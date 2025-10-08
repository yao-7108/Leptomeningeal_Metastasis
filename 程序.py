# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tabpfn import TabPFNClassifier
import warnings

# 过滤警告
warnings.filterwarnings('ignore', category=UserWarning)

# Page configuration
st.set_page_config(
    page_title="Leptomeningeal Metastasis", 
    layout="centered",
    page_icon="🌲"
)

st.title("High-Risk Prediction for LUAD Patients with LM")
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

# 1. Load model
@st.cache_resource
def load_model(path="tabpfn.joblib"):
    return joblib.load(path)

model = load_model()

# 2. User inputs
st.sidebar.header("Input Parameters")

# Use column layout for sidebar
col1 = st.sidebar.columns(1)[0]

CO2 = col1.slider("Co2", 2.51, 30.7, 21.75)
Ca = col1.slider("Ca", 0.89, 102.1, 2.02)
EGFR_ex21_L858R = col1.slider("Egfr Ex21 L858R", 0, 1, 0)
KPS = col1.slider("Kps", 60, 90, 72)
PFS = col1.slider("Pfs", 0, 94, 12)
IBIL = col1.slider("Ibil", 1.1, 30.5, 13.11)

# 3. Create feature DataFrame with proper column names
feature_names = ['PFS', 'EGFR ex21 L858R', 'KPS', 'IBIL', 'Ca', 'CO2']
features = pd.DataFrame([[PFS, EGFR_ex21_L858R, KPS, IBIL, Ca, CO2]], 
                       columns=feature_names)

# 4. Prediction button
predict_button = st.sidebar.button("Run Prediction", use_container_width=True, type="primary")

# 5. Result display
if predict_button:
    st.divider()
    st.subheader("📊 Prediction Results")
    
    try:
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class labels
        class_labels = model.classes_
        
        # Result cards
        col1, col2 = st.columns(2)
        
        with col1:
            result = 'Diseased' if prediction == 1 else 'Non-diseased'
            st.metric("Prediction", result)
            
        with col2:
            risk_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            st.metric("Disease Probability", f"{risk_prob:.2%}")
        
        # Probability distribution visualization
        st.subheader("Probability Distribution")
        proba_df = pd.DataFrame({
            'Class': ['Non-diseased', 'Diseased'],
            'Probability': probabilities
        })
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#1f77b4', '#ff7f0e']
        bars = ax.bar(proba_df['Class'], proba_df['Probability'], color=colors)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Predicted Probability Distribution', fontsize=14)
        plt.xticks(fontsize=11)
        st.pyplot(fig)
        
        # SHAP plots
        st.subheader("Model Explanation")
        try:
            shap_1 = Image.open('shap_特征重要度.png')
            shap_2 = Image.open('shap.png')
            col1, col2 = st.columns(2)
            with col1:
                st.image(shap_1, caption='SHAP Feature Importance')
            with col2:
                st.image(shap_2, caption='SHAP Summary Plot')
        except FileNotFoundError:
            st.info("SHAP plot images not found. Please ensure 'shap_特征重要度.png' and 'shap.png' are in the correct directory.")

        # Feature details
        with st.expander("📋 Input Feature Details"):
            st.write("Current input feature values:")
            feature_table = pd.DataFrame({
                'Feature': feature_names,
                'Value': features.iloc[0].values
            })
            st.table(feature_table)
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Please check if model and input features match")

# 6. Instructions
st.divider()
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    **Model Information:**
    - Model Type: TabPFN Classifier
    - Input Features: 6 clinical parameters
    - Output Classes: 2 (Non-diseased vs Diseased)
    
    **User Guide:**
    1. Adjust feature parameters in the left sidebar
    2. Click "Run Prediction" button to get results
    3. View prediction results and probability distribution
    
    **Feature Explanation:**
    - **PFS**: Progression-Free Survival (months)
    - **EGFR_ex21_L858R**: EGFR Exon 21 L858R Mutation (0=No, 1=Yes)
    - **KPS**: Karnofsky Performance Status score
    - **IBIL**: Indirect Bilirubin level
    - **Ca**: Calcium level
    - **CO2**: Carbon Dioxide level
    
    **Notes:**
    - All input features must match the model training format
    - Continuous values use sliders
    - Categorical features use 0/1 encoding
    """)

# 7. Footer
st.caption("© 2025 Leptomeningeal Metastasis Predict")
st.caption("Original paper: Machine Learning for Predicting Leptomeningeal Metastasis and Prognosis in Lung Adenocarcinoma: a multi-center retrospective study Using the \"Prompt\" Model.")