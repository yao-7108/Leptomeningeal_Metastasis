# app.py
import streamlit as st
import joblib
import numpy as np

# 页面基本配置
st.set_page_config(page_title="RFC 演示", layout="centered")

st.title("随机森林分类器 演示")

# 1. 加载模型
@st.cache(allow_output_mutation=True)
def load_model(path="rfc_model.joblib"):
    return joblib.load(path)

model = load_model()

# 2. 用户输入
st.sidebar.header("输入参数")
# 以 Iris 数据集为例，这里设定四个滑块
sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 5.8)
sepal_width  = st.sidebar.slider("Sepal width",  2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 4.35)
petal_width  = st.sidebar.slider("Petal width",  0.1, 2.5, 1.3)

input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# 3. 预测并显示结果
if st.sidebar.button("预测"):
    pred = model.predict(input_features)[0]
    proba = model.predict_proba(input_features).max()
    st.write(f"**预测类别：** {pred}")
    st.write(f"**预测概率：** {proba:.2f}")
