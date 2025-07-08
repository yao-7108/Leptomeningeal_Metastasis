# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# 页面基本配置
st.set_page_config(
    page_title="Leptomeningeal Metastasis", 
    layout="centered",
    page_icon="🌲"
)

st.title("随机森林分类器 演示")
st.markdown("""
<style>
div[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.stMetric {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# 1. 加载模型
@st.cache_resource
def load_model(path="rfc.joblib"):
    return joblib.load(path)

model = load_model()

# 2. 用户输入
st.sidebar.header("输入参数")

# 使用列布局优化侧边栏
col1, col2 = st.sidebar.columns(2)

with col1:
    Gender = st.sidebar.slider("Gender", 0, 1, 1)
    Age = st.sidebar.slider("Age", 31, 95, 57)
    Smoking = st.sidebar.slider("Smoking", 0, 1, 0)
    PFS = st.sidebar.slider("Pfs", 0, 94, 12)
    EGFR_ex21_L858R = st.sidebar.slider("Egfr Ex21 L858R", 0, 1, 0)
    rd_generation_TKIs = st.sidebar.slider("3Rd-Generation Tkis", 0, 1, 0)
    Bevacizumab = st.sidebar.slider("Bevacizumab", 0, 1, 0)
    Metastatic_Organs = st.sidebar.slider("Metastatic Organs", 0, 3, 2)
    Dizziness = st.sidebar.slider("Dizziness", 0, 1, 0)
    KPS = st.sidebar.slider("Kps", 60, 90, 72)
    ALB = st.sidebar.slider("Alb", 3.3, 51.3, 42.72)

with col2:
    GLB = st.sidebar.slider("Glb", 15.4, 47.5, 24.54)
    AG = st.sidebar.slider("A/G", 0.9, 30.5, 1.75)
    TBIL = st.sidebar.slider("Tbil", 3.1, 36.2, 15.46)
    DBIL = st.sidebar.slider("Dbil", 0.1, 17.2, 2.34)
    IBIL = st.sidebar.slider("Ibil", 1.1, 30.5, 13.11)
    ALT = st.sidebar.slider("Alt", 2, 128, 16)
    GLU = st.sidebar.slider("Glu", 3.06, 277.0, 5.34)
    K = st.sidebar.slider("K", 2.95, 8.06, 4.18)
    Ca = st.sidebar.slider("Ca", 0.89, 102.1, 2.02)
    CO2 = st.sidebar.slider("Co2", 2.51, 30.7, 21.75)
    PCA1 = st.sidebar.slider("Pca1", -103.9169180495229, 773.7879743514883, -18.68)
    PCA2 = st.sidebar.slider("Pca2", -203.9109840184679, 232.96592983059233, -76.81)

# 3. 创建特征数组
features = np.array([
    Gender, Age, Smoking, PFS, EGFR_ex21_L858R, 
    rd_generation_TKIs, Bevacizumab, Metastatic_Organs, 
    Dizziness, KPS, ALB, GLB, AG, TBIL, DBIL, IBIL, 
    ALT, GLU, K, Ca, CO2, PCA1, PCA2
]).reshape(1, -1)

# 4. 预测按钮
predict_button = st.sidebar.button("运行预测", use_container_width=True, type="primary")

# 5. 结果展示区域
if predict_button:
    st.divider()
    st.subheader("📊 预测结果")
    
    try:
        # 执行预测
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        #中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 获取类别标签（假设为0和1）
        class_labels = model.classes_
        
        # 结果卡片布局
        col1, col2= st.columns(2)
        
        with col1:
            result = '患病' if prediction == 1 else '未患病'
            st.metric("预测结果", result)
            
        with col2:
            max_prob = max(probabilities)
            st.metric("预测概率", f"{max_prob:.2%}")
        
        # 概率分布可视化
        st.subheader("概率分布")
        proba_df = pd.DataFrame({
            '类别': class_labels,
            '概率': probabilities
        })
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(proba_df['类别'].astype(str), proba_df['概率'], color=['#1f77b4', '#ff7f0e'])
        ax.set_xticks([0,1])
        ax.set_xticklabels(['未患病', '患病'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('prob')
        ax.set_title('Predicted probability for each category')
        st.pyplot(fig)
        
        # # 特征重要性分析（如果模型支持）
        # if hasattr(model, 'feature_importances_'):
        #     st.subheader("特征重要性")
            
            # 获取特征名称（根据您的输入顺序）
        feature_names = [
            "Gender", "Age", "Smoking", "PFS", "EGFR_ex21_L858R", 
            "3rd_generation_TKIs", "Bevacizumab", "Metastatic_Organs", 
            "Dizziness", "KPS", "ALB", "GLB", "A/G", "TBIL", "DBIL", "IBIL", 
            "ALT", "GLU", "K", "Ca", "CO2", "PCA1", "PCA2"
        ]
            
        #     # 创建特征重要性数据框
        #     importance_df = pd.DataFrame({
        #         '特征': feature_names,
        #         '重要性': model.feature_importances_
        #     }).sort_values('重要性', ascending=False)
            
        #     # 只显示前10个重要特征
        #     # top_features = importance_df.head(10)
        #     top_features = importance_df
            
        #     # 创建水平条形图
        #     fig2, ax2 = plt.subplots(figsize=(10, 6))
        #     ax2.barh(top_features['特征'][::-1], top_features['重要性'][::-1], color='#2ca02c')
        #     ax2.set_xlabel('Importance')
        #     ax2.set_title('Top 10 Important Features')
        #     st.pyplot(fig2)
        st.subheader("SHAP图")
        #读取shap_特征重要度.png以及shap.png两张图片进行展示
        shap_1 = Image.open('shap_特征重要度.png')
        shap_2 = Image.open('shap.png')
        col1, col2 = st.columns(2)
        with col1:
            st.image(shap_1, caption='SHAP特征重要度')
        with col2:
            st.image(shap_2, caption='SHAP图')


        
        # 原始特征值展示
        with st.expander("📋 输入特征详情"):
            st.write("当前输入的特征值:")
            feature_table = pd.DataFrame({
                '特征': feature_names,
                '值': features[0]
            })
            st.table(feature_table)
            
    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        st.error("请检查模型和输入特征是否匹配")

# 6. 添加说明
st.divider()
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    **模型信息:**
    - 模型类型: 随机森林分类器 (RFC)
    - 输入特征数: 23
    - 输出类别数: 2
    
    **操作指南:**
    1. 在左侧边栏调整各特征参数
    2. 点击"运行预测"按钮获取结果
    3. 查看预测结果和特征重要性分析
    
    **特征解释:**
    - **Gender**: 性别 (0=女, 1=男)
    - **Smoking**: 吸烟史 (0=无, 1=有)
    - **KPS**: Karnofsky功能状态评分
    - **PCA1/PCA2**: 主成分分析结果
    - **其他**: 临床检测指标
    
    **注意事项:**
    - 所有输入特征需与模型训练时一致
    - 连续值特征使用滑动条调整
    - 分类特征使用0/1表示
    """)

# 7. 页脚
st.caption("© 2025 Leptomeningeal Metastasis Predict \n 原文：Machine Learning for Predicting Leptomeningeal Metastasis and Prognosis in Lung Adenocarcinoma: a multi-center retrospective study Using the \"Prompt\" Model")
