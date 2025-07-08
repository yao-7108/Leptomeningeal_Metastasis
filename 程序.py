# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 页面基本配置
st.set_page_config(
    page_title="RFC 演示", 
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
    Age = st.sidebar.slider("Age", 31, 95, 95)
    Smoking = st.sidebar.slider("Smoking", 0, 1, 1)
    PFS = st.sidebar.slider("Pfs", 0, 94, 94)
    EGFR_ex21_L858R = st.sidebar.slider("Egfr Ex21 L858R", 0, 1, 1)
    rd_generation_TKIs = st.sidebar.slider("3Rd-Generation Tkis", 0, 1, 1)
    Bevacizumab = st.sidebar.slider("Bevacizumab", 0, 1, 1)
    Metastatic_Organs = st.sidebar.slider("Metastatic Organs", 0, 3, 3)
    Dizziness = st.sidebar.slider("Dizziness", 0, 1, 1)
    KPS = st.sidebar.slider("Kps", 60, 90, 90)
    ALB = st.sidebar.slider("Alb", 3.3, 51.3, 51.3)

with col2:
    GLB = st.sidebar.slider("Glb", 15.4, 47.5, 47.5)
    AG = st.sidebar.slider("A/G", 0.9, 30.5, 30.5)
    TBIL = st.sidebar.slider("Tbil", 3.1, 36.2, 36.2)
    DBIL = st.sidebar.slider("Dbil", 0.1, 17.2, 17.2)
    IBIL = st.sidebar.slider("Ibil", 1.1, 30.5, 30.5)
    ALT = st.sidebar.slider("Alt", 2, 128, 128)
    GLU = st.sidebar.slider("Glu", 3.06, 277.0, 277.0)
    K = st.sidebar.slider("K", 2.95, 8.06, 8.06)
    Ca = st.sidebar.slider("Ca", 0.89, 102.1, 102.1)
    CO2 = st.sidebar.slider("Co2", 2.51, 30.7, 30.7)
    PCA1 = st.sidebar.slider("Pca1", -103.9169180495229, 773.7879743514883, 773.7879743514883)
    PCA2 = st.sidebar.slider("Pca2", -203.9109840184679, 232.96592983059233, 232.96592983059233)

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
        
        # 获取类别标签（假设为0和1）
        class_labels = model.classes_
        
        # 结果卡片布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("预测类别", f"类别 {prediction}", delta="预测结果")
            
        with col2:
            max_prob = max(probabilities)
            st.metric("最高概率", f"{max_prob:.2%}", delta="置信度")
        
        # 概率分布可视化
        st.subheader("类别概率分布")
        proba_df = pd.DataFrame({
            '类别': class_labels,
            '概率': probabilities
        })
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(proba_df['类别'].astype(str), proba_df['概率'], color=['#1f77b4', '#ff7f0e'])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('概率')
        ax.set_title('各类别预测概率')
        st.pyplot(fig)
        
        # 特征重要性分析（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            st.subheader("特征重要性")
            
            # 获取特征名称（根据您的输入顺序）
            feature_names = [
                "Gender", "Age", "Smoking", "PFS", "EGFR_ex21_L858R", 
                "rd_generation_TKIs", "Bevacizumab", "Metastatic_Organs", 
                "Dizziness", "KPS", "ALB", "GLB", "A/G", "TBIL", "DBIL", "IBIL", 
                "ALT", "GLU", "K", "Ca", "CO2", "PCA1", "PCA2"
            ]
            
            # 创建特征重要性数据框
            importance_df = pd.DataFrame({
                '特征': feature_names,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            # 只显示前10个重要特征
            top_features = importance_df.head(10)
            
            # 创建水平条形图
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.barh(top_features['特征'], top_features['重要性'], color='#2ca02c')
            ax2.set_xlabel('重要性')
            ax2.set_title('Top 10 重要特征')
            st.pyplot(fig2)
        
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
st.caption("© 2023 随机森林分类器演示 | 使用Streamlit构建")