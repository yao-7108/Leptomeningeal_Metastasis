#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import warnings
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload
from typing_extensions import Self, override

import einops
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabpfn.architectures.base.encoders import (
    LinearInputEncoderStep,
    NanHandlingEncoderStep,
    SequentialEncoder,
)
from tabpfn.architectures.base.layer import PerFeatureEncoderLayer
from tabpfn.architectures.interface import Architecture

if TYPE_CHECKING:
    from tabpfn.architectures.base.config import ModelConfig


@contextmanager
def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)


class LayerStack(nn.Module):
    """Similar to nn.Sequential, but with layer dropout."""

    def __init__(
        self,
        *,
        layers: Iterable[nn.Module],
        recompute_each_layer: bool,
        min_num_layers_layer_dropout: int | None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else len(self.layers)
        )
        self.recompute_each_layer = recompute_each_layer

    @classmethod
    def of_repeated_layer(
        cls,
        layer_creator: Callable[[], nn.Module],
        *,
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ) -> Self:
        """Returns an instance containing the given layer repeated num_layers times."""
        return cls(
            layers=[layer_creator() for _ in range(num_layers)],
            recompute_each_layer=recompute_each_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        n_layers = torch.randint(
            low=self.min_num_layers_layer_dropout, high=len(self.layers) + 1, size=(1,)
        ).item()

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)  # type: ignore
            else:
                x = layer(x, **kwargs)

        return x


class PerFeatureTransformer(Architecture):
    """A Transformer model processes a token per feature and sample.

    This model extends the standard Transformer architecture to operate on a
    per-feature basis.
    It allows for processing each feature separately while still leveraging the
    power of self-attention.

    The model consists of an encoder, decoder, and optional components such
    as a feature positional embedding and a separate decoder for each feature.
    """

    def __init__(  # noqa: D417, PLR0913
        self,
        *,
        config: ModelConfig,
        encoder: nn.Module | None = None,
        y_encoder: nn.Module | None = None,
        n_out: int = 1,
        activation: Literal["gelu", "relu"] = "gelu",
        min_num_layers_layer_dropout: int | None = None,
        zero_init: bool = True,
        nlayers_decoder: int | None = None,
        use_encoder_compression_layer: bool = False,
        precomputed_kv: (
            list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        cache_trainset_representation: bool = False,
        # TODO: List explicitly
        **layer_kwargs: Any,
    ):
        """Initializes the PerFeatureTransformer module.

        Args:
            encoder:
                Pass a nn.Module that takes in a batch of sequences of inputs and
                returns something of the shape (seq_len, batch_size, ninp)
            y_encoder:
                A nn.Module that takes in a batch of sequences of outputs and
                returns something of the shape (seq_len, batch_size, ninp)
            activation: An activation function, "gelu" or "relu"
            min_num_layers_layer_dropout:
                If this is set, it enables to drop the last
                layers randomly during training up to this number.
            feature_positional_embedding:
                There is a risk that our models confuse
                features with each other. This positional embedding is added to the
                features to help the model distinguish them.
                We recommend setting this to "subspace".
            zero_init:
                If True, the last sublayer of each attention and MLP layer will
                be initialized with zeros.
                Thus, the layers will start out as identity functions.
            nlayers_decoder:
                If ModelConfig.use_separate_decoder is True, must be set to specify the
                number of layers in the decoder.
            use_encoder_compression_layer: Experimental
            precomputed_kv: Experimental
            layer_kwargs:
                TODO: document.
                for now have a look at layer.py:PerFeatureEncoderLayer.
        """
        super().__init__()

        if encoder is None:
            encoder = SequentialEncoder(
                LinearInputEncoderStep(
                    num_features=1,
                    emsize=config.emsize,
                    replace_nan_by_zero=False,
                    bias=True,
                    in_keys=("main",),
                    out_keys=("output",),
                ),
            )

        if y_encoder is None:
            y_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,
                    emsize=config.emsize,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            )
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.ninp = config.emsize
        self.nhid_factor = config.nhid_factor
        nhid = self.ninp * self.nhid_factor
        self.features_per_group = config.features_per_group
        self.cache_trainset_representation = cache_trainset_representation
        self.cached_embeddings: torch.Tensor | None = None

        layer_creator = lambda: PerFeatureEncoderLayer(
            config=config,
            dim_feedforward=nhid,
            activation=activation,
            zero_init=zero_init,
            precomputed_kv=(
                precomputed_kv.pop(0) if precomputed_kv is not None else None
            ),
            **layer_kwargs,
        )

        self.transformer_encoder = LayerStack.of_repeated_layer(
            layer_creator=layer_creator,
            num_layers=config.nlayers,
            recompute_each_layer=config.recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )

        self.transformer_decoder = None
        if config.use_separate_decoder:
            if nlayers_decoder is None:
                raise ValueError(
                    "nlayers_decoder must be specified "
                    "if ModelConfig.use_separate_decoder is True."
                )
            self.transformer_decoder = LayerStack.of_repeated_layer(
                layer_creator=layer_creator,
                num_layers=nlayers_decoder,
            )

        self.global_att_embeddings_for_compression = None
        if use_encoder_compression_layer:
            assert config.use_separate_decoder
            num_global_att_tokens_for_compression = 512

            self.global_att_embeddings_for_compression = nn.Embedding(
                num_global_att_tokens_for_compression,
                self.ninp,
            )

            self.encoder_compression_layer = LayerStack.of_repeated_layer(
                layer_creator=layer_creator,
                num_layers=2,
            )

        self.n_out = n_out
        self.decoder_dict = nn.ModuleDict(
            {
                "standard": nn.Sequential(
                    nn.Linear(self.ninp, nhid),
                    nn.GELU(),
                    nn.Linear(nhid, n_out),
                )
            }
        )

        self.feature_positional_embedding = config.feature_positional_embedding
        if self.feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(
                1_000, self.ninp
            )
        elif self.feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(
                self.ninp // 4, self.ninp
            )

        self.dag_pos_enc_dim = config.dag_pos_enc_dim
        self.cached_feature_positional_embeddings: torch.Tensor | None = None
        self.seed = config.seed

    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """Sets the save_peak_mem_factor for all layers.

        This factor controls how much memory is saved during the forward pass
        in inference mode.

        Setting this factor > 1 will cause the model to save more memory during
        the forward pass in inference mode.

        A value of 8 is good for a 4x larger width in the fully-connected layers.
        and yields a situation were we need around
        `2*num_features*num_items*emsize*2` bytes of memory

        for a forward pass (using mixed precision).

        WARNING: It should only be used with post-norm.

        Args:
            factor: The save_peak_mem_factor to set. Recommended value is 8.
        """
        for layer in self.transformer_encoder.layers:
            assert hasattr(
                layer,
                "save_peak_mem_factor",
            ), "Layer does not have save_peak_mem_factor"
            layer.save_peak_mem_factor = factor  # type: ignore

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("features_per_group", 1)
        state.setdefault("feature_positional_embedding", None)
        super().__setstate__(state)

    @overload
    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: Literal[True] = True,
        categorical_inds: list[list[int]] | None = None,
        style: torch.Tensor | None = None,
        data_dags: list[nx.DiGraph] | None = None,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: Literal[False],
        categorical_inds: list[list[int]] | None = None,
        style: torch.Tensor | None = None,
        data_dags: list[nx.DiGraph] | None = None,
    ) -> dict[str, torch.Tensor]: ...

    @override
    def forward(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        style: torch.Tensor | None = None,
        data_dags: list[nx.DiGraph] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Perform a forward pass.

        See ModelInterface.forward() for the full docstring.

        Args:
            x: The input data. Either:
                - A Tensor with shape
                  `[(train+test) rows, batch size, num input features]`.
                - A dictionary containing at least `{"main": x}`, where `x` is the
                  Tensor above. The dictionary may also contain additional keys, which
                  are relevant for particular encoders.
            y: The target data. Either:
                - A Tensor with shape `(train rows)`, `(train_rows, batch_size)`, or
                  shape `(train_rows, batch_size, 1)`.
                - A dictionary containing at least `{"main": y}`, where `y` is the
                  Tensor above. The dictionary may also contain additional keys, which
                  are relevant for particular encoders.
                - `None`, if there are no training rows, as when making predictions
                  using the KV cache.
            only_return_standard_out: Whether to only return the standard output.
            categorical_inds: The indices of categorical features.
            style: The style vector.
            data_dags: The data DAGs for each example in the batch.
        """
        assert style is None

        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:
            x = {"main": x}
        seq_len, batch_size, num_features = x["main"].shape

        if isinstance(y, dict):
            assert "main" in set(y.keys()), f"Main must be in input keys: {y.keys()}."
        else:
            y = {"main": y}

        training_targets_provided = y["main"] is not None and y["main"].shape[0] > 0
        if not training_targets_provided and not self.cache_trainset_representation:
            raise ValueError(
                "If caching the training data representation is disabled, "
                "then you must provide some training labels"
            )

        if y["main"] is None:
            # TODO: check dtype.
            y["main"] = torch.zeros(
                0,
                batch_size,
                device=x["main"].device,
                dtype=x["main"].dtype,
            )

        # The model will make predictions from the single_eval_pos'th row onwards. We
        # want to start making predictions at the first row without target data.
        single_eval_pos = y["main"].shape[0]

        for k in x:
            num_features_ = x[k].shape[2]

            # pad to multiple of features_per_group
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )

        # Splits up the input into subgroups
        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "s b (f n) -> b s f n",
                n=self.features_per_group,
            )  # s b f -> b s #groups #features_per_group

        # We have to re-work categoricals based on the subgroup they fall into.
        categorical_inds_to_use: list[list[list[int]]] | None = None
        if categorical_inds is not None:
            assert isinstance(
                categorical_inds[0],
                list,
            ), "categorical_inds must be a list of lists (one per batch item)"

            new_categorical_inds = []
            n_subgroups = x["main"].shape[2]

            # For each batch item
            for batch_idx in range(batch_size):
                # For each subgroup
                for subgroup in range(n_subgroups):
                    subgroup_lower = subgroup * self.features_per_group
                    subgroup_upper = (subgroup + 1) * self.features_per_group
                    subgroup_indices = [
                        i - subgroup_lower
                        for i in categorical_inds[batch_idx]
                        if subgroup_lower <= i < subgroup_upper
                    ]
                    # Add this subgroup's indices to the flattened list
                    new_categorical_inds.append(subgroup_indices)

            categorical_inds_to_use = new_categorical_inds

        for k in y:
            if y[k].ndim == 1:
                y[k] = y[k].unsqueeze(-1)
            if y[k].ndim == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert k != "main" or y[k].shape[1] == single_eval_pos, (
                    "For main y, y must not be given for target"
                    " time steps (Otherwise the solution is leaked)."
                )
                if y[k].shape[1] == single_eval_pos:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

        # making sure no label leakage ever happens
        y["main"][single_eval_pos:] = torch.nan

        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del y
        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

        extra_encoders_args = {}
        if categorical_inds_to_use is not None and isinstance(
            self.encoder,
            SequentialEncoder,
        ):
            # Transform cat. features accordingly to correspond to following to merge
            # of batch and feature_group dimensions below (i.e., concat lists)
            extra_encoders_args["categorical_inds"] = sum(categorical_inds_to_use, [])  # noqa: RUF017

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")
        embedded_x = einops.rearrange(
            self.encoder(
                x,
                single_eval_pos=single_eval_pos,
                cache_trainset_representation=self.cache_trainset_representation,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x,
            embedded_y,
            data_dags=data_dags,
            num_features=num_features,
            seq_len=seq_len,
            cache_embeddings=(
                self.cache_trainset_representation and training_targets_provided
            ),
            use_cached_embeddings=(
                self.cache_trainset_representation and not training_targets_provided
            ),
        )
        del data_dags

        # b s f e + b s 1 e -> b s f+1 e
        embedded_input = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x and y."
                "Check that you do not feed NaNs or use a NaN-handling enocder."
                "Your embedded x and y returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_y).any()=}",
            )
        del embedded_y, embedded_x

        encoder_out = self.transformer_encoder(
            (
                embedded_input
                if not self.transformer_decoder
                else embedded_input[:, :single_eval_pos]
            ),
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # b s f+1 e -> b s f+1 e

        # If we are using a decoder
        if self.transformer_decoder:
            assert encoder_out.shape[1] == single_eval_pos

            if self.global_att_embeddings_for_compression is not None:
                # TODO: fixed number of compression tokens
                train_encoder_out = self.encoder_compression_layer(
                    self.global_att_embeddings_for_compression,
                    att_src=encoder_out[:, single_eval_pos],
                    single_eval_pos=single_eval_pos,
                )

            test_encoder_out = self.transformer_decoder(
                embedded_input[:, single_eval_pos:],
                single_eval_pos=0,
                att_src=encoder_out,
            )
            encoder_out = torch.cat([encoder_out, test_encoder_out], 1)
            del test_encoder_out

        del embedded_input

        # out: s b e
        test_encoder_out = encoder_out[:, single_eval_pos:, -1].transpose(0, 1)

        if only_return_standard_out:
            assert self.decoder_dict is not None
            output_decoded = self.decoder_dict["standard"](test_encoder_out)
        else:
            output_decoded = (
                {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
                if self.decoder_dict is not None
                else {}
            )

            # out: s b e
            train_encoder_out = encoder_out[:, :single_eval_pos, -1].transpose(0, 1)
            output_decoded["train_embeddings"] = train_encoder_out
            output_decoded["test_embeddings"] = test_encoder_out

        return output_decoded

    def add_embeddings(  # noqa: C901, PLR0912
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        data_dags: Iterable[nx.DiGraph] | None,
        num_features: int,
        seq_len: int,
        cache_embeddings: bool = False,
        use_cached_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_cached_embeddings and self.cached_embeddings is not None:
            assert (
                data_dags is None
            ), "Caching embeddings is not supported with data_dags at this point."
            x += self.cached_embeddings[None, None]
            return x, y

        # TODO: we should probably hardcode the seed here
        # I think we never want to change it?
        with isolate_torch_rng(self.seed, device=x.device):
            if self.feature_positional_embedding == "normal_rand_vec":
                embs = torch.randn(
                    (x.shape[2], x.shape[3]),
                    device=x.device,
                    dtype=x.dtype,
                )
                x += embs[None, None]
            elif self.feature_positional_embedding == "uni_rand_vec":
                embs = (
                    torch.rand(
                        (x.shape[2], x.shape[3]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    * 2
                    - 1
                )
                x += embs[None, None]
            elif self.feature_positional_embedding == "learned":
                w = self.feature_positional_embedding_embeddings.weight
                embs = w[
                    torch.randint(
                        0,
                        w.shape[0],
                        (x.shape[2],),
                    )
                ]
                x += embs[None, None]
            elif self.feature_positional_embedding == "subspace":
                embs = torch.randn(
                    (x.shape[2], x.shape[3] // 4),
                    device=x.device,
                    dtype=x.dtype,
                )
                embs = self.feature_positional_embedding_embeddings(embs)
                x += embs[None, None]
            elif self.feature_positional_embedding is None:
                embs = None
            else:
                raise ValueError(f"Unknown {self.feature_positional_embedding=}")

        self.cached_embeddings = None
        if cache_embeddings and embs is not None:
            assert (
                data_dags is None
            ), "Caching embeddings is not supported with data_dags at this point."
            self.cached_embeddings = embs

        # TODO(old) should this go into encoder?
        # could also be made a bit more concise by moving down to operate on full_x
        if data_dags is not None:
            for b_i, data_dag in enumerate(data_dags):
                # TODO(eddibergman): Very inneficient way to make a full connect
                # DiGraph
                g_: nx.DiGraph = data_dag.copy()
                while _networkx_add_direct_connections(g_):
                    pass

                subgraph: nx.DiGraph = g_.subgraph(  # type: ignore
                    [
                        n
                        for n, info in g_.nodes.items()
                        if (info["is_feature"] or info["is_target"])
                    ],
                )
                assert self.dag_pos_enc_dim is not None
                k = self.dag_pos_enc_dim
                assert k > 0
                _add_pos_emb(subgraph, k=k)

                graph_pos_embs_features = torch.zeros((num_features, k))
                graph_pos_embs_targets = torch.zeros((1, k))  # shape: (num_targets, k)

                for node_info in subgraph.nodes.values():
                    for feature_idx in node_info.get("feature_idxs", []):
                        graph_pos_embs_features[feature_idx] = node_info[
                            "positional_encoding"
                        ]
                    for target_idx in node_info.get("target_idxs", []):
                        graph_pos_embs_targets[target_idx] = node_info[
                            "positional_encoding"
                        ]

                graph_pos_embs_targets -= graph_pos_embs_features.mean(0, keepdim=True)
                graph_pos_embs_features -= graph_pos_embs_features.mean(0, keepdim=True)

                graph_pos_embs_features = graph_pos_embs_features[None].expand(
                    seq_len,
                    -1,
                    -1,
                )
                x[b_i, :, :, :k] += graph_pos_embs_features.to(y.device, y.dtype)

                graph_pos_embs_targets = (
                    graph_pos_embs_targets[None].expand(seq_len, -1, -1).squeeze(-2)
                )
                y[b_i, :, :k] += graph_pos_embs_targets.to(y.device, y.dtype)
        else:
            assert self.dag_pos_enc_dim is None or self.dag_pos_enc_dim == 0

        return x, y

    def empty_trainset_representation_cache(self) -> None:
        """Clears any cached training data in the model.

        e.g. the key-value cache of each transformer layer.
        """
        for layer in (self.transformer_decoder or self.transformer_encoder).layers:
            layer.empty_trainset_representation_cache()

    def _transform_categorical_indices_feat_groups(
        self, categorical_inds: list[int], n_subgroups: int
    ) -> list[list[int]]:
        """Transform the categorical indices list(s)
        to align with the feature groups.

        Args:
            categorical_inds: categorical indices as 2D list
            n_subgroups: number of subgroups.
        """
        new_categorical_inds = []
        for subgroup in range(n_subgroups):
            subgroup_lower = subgroup * self.features_per_group
            subgroup_upper = (subgroup + 1) * self.features_per_group
            subgroup_indices = [
                i - subgroup_lower
                for i in categorical_inds
                if subgroup_lower <= i < subgroup_upper
            ]
            new_categorical_inds.append(subgroup_indices)
        return new_categorical_inds


def _networkx_add_direct_connections(graph: nx.DiGraph) -> bool:
    added_connection = False
    # Get the list of nodes
    nodes = list(graph.nodes)

    # Iterate over each node
    for node in nodes:
        # Get the direct neighbors of the current node
        neighbors = list(graph.neighbors(node))

        # Iterate over the neighbors of the current node
        for neighbor in neighbors:
            # Get the neighbors of the neighbor
            second_neighbors = list(graph.neighbors(neighbor))

            # Iterate over the neighbors of the neighbor
            for second_neighbor in second_neighbors:
                # Add a direct edge from the current node to the second neighbor,
                # if it doesn't exist already
                if second_neighbor not in graph.neighbors(node):
                    graph.add_edge(node, second_neighbor)

                    added_connection = True
    return added_connection


def _add_pos_emb(
    graph: nx.DiGraph,
    *,
    is_undirected: bool = False,
    k: int = 20,
) -> None:
    from scipy.sparse.linalg import eigs, eigsh

    eig_fn = eigs if not is_undirected else eigsh

    L = nx.directed_laplacian_matrix(graph)
    np.nan_to_num(L, nan=0.0, copy=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eig_vals, eig_vecs = eig_fn(  # type: ignore
            L,
            k=k + 1,
            which="SR" if not is_undirected else "SA",
            return_eigenvectors=True,
        )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe_ = torch.from_numpy(eig_vecs[:, 1 : k + 1])
        pe = torch.zeros(len(eig_vecs), k)
        pe[:, : pe_.shape[1]] = pe_
        sign = -1 + 2 * torch.randint(0, 2, (k,))
        pe *= sign

        # TODO(old) Double check the ordering is right
        for n, pe_ in zip(graph.nodes(), pe):
            graph.nodes[n]["positional_encoding"] = pe_
