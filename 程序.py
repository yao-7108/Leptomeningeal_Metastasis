# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Leptomeningeal Metastasis", 
    layout="centered",
    page_icon="🌲"
)

st.title("Hiqh-Risk Prediction for LUAD Patients with LM")
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

# 3. Create feature array
features = np.array([
    Gender, Age, Smoking, PFS, EGFR_ex21_L858R, 
    rd_generation_TKIs, Bevacizumab, Metastatic_Organs, 
    Dizziness, KPS, ALB, GLB, AG, TBIL, DBIL, IBIL, 
    ALT, GLU, K, Ca, CO2, PCA1, PCA2
]).reshape(1, -1)

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
            max_prob = max(probabilities)
            st.metric("Probability", f"{max_prob:.2%}")
        
        # Probability distribution visualization
        st.subheader("Probability Distribution")
        proba_df = pd.DataFrame({
            'Class': class_labels,
            'Probability': probabilities
        })
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(proba_df['Class'].astype(str), proba_df['Probability'], 
                      color=['#1f77b4', '#ff7f0e'])
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Non-diseased', 'Diseased'])
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Predicted Probability Distribution')
        st.pyplot(fig)
        
        # SHAP plots
        st.subheader("SHAP Plots")
        shap_1 = Image.open('shap_特征重要度.png')
        shap_2 = Image.open('shap.png')
        col1, col2 = st.columns(2)
        with col1:
            st.image(shap_1, caption='SHAP Feature Importance')
        with col2:
            st.image(shap_2, caption='SHAP Summary Plot')

        # Feature details
        with st.expander("📋 Input Feature Details"):
            st.write("Current input feature values:")
            feature_names = [
                "Gender", "Age", "Smoking", "PFS", "EGFR_ex21_L858R", 
                "3rd_generation_TKIs", "Bevacizumab", "Metastatic_Organs", 
                "Dizziness", "KPS", "ALB", "GLB", "A/G", "TBIL", "DBIL", "IBIL", 
                "ALT", "GLU", "K", "Ca", "CO2", "PCA1", "PCA2"
            ]
            feature_table = pd.DataFrame({
                'Feature': feature_names,
                'Value': features[0]
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
    - Model Type: Random Forest Classifier (RFC)
    - Input Features: 23
    - Output Classes: 2
    
    **User Guide:**
    1. Adjust feature parameters in the left sidebar
    2. Click "Run Prediction" button to get results
    3. View prediction results and feature analysis
    
    **Feature Explanation:**
    - **Gender**: Gender (0=Female, 1=Male)
    - **Smoking**: Smoking history (0=No, 1=Yes)
    - **KPS**: Karnofsky Performance Status score
    - **PCA1/PCA2**: Principal Component Analysis results
    - **Others**: Clinical test indicators
    
    **Notes:**
    - All input features must match the model training format
    - Continuous values use sliders
    - Categorical features use 0/1 encoding
    """)

# 7. Footer
st.caption("© 2025 Leptomeningeal Metastasis Predict")
st.caption("Original paper: Machine Learning for Predicting Leptomeningeal Metastasis and Prognosis in Lung Adenocarcinoma: a multi-center retrospective study Using the \"Prompt\" Model.")