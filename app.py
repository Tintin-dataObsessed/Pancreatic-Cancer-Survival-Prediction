import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("joblib")
install("scikit-learn")

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Configuration
st.set_page_config(
    page_title="Pancreatic Cancer Survival Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom model loader
class ModelLoader:
    @staticmethod
    def load_scaler(filepath):
        """Load and validate the StandardScaler"""
        try:
            scaler = joblib.load(filepath)
            required_attrs = ['mean_', 'scale_', 'feature_names_in_', 'n_features_in_']
            if not all(hasattr(scaler, attr) for attr in required_attrs):
                raise ValueError("Loaded scaler is missing required attributes")
            return scaler
        except Exception as e:
            st.error(f"Scaler loading failed: {str(e)}")
            return None

    @staticmethod
    def load_encoder(filepath):
        """Load and validate the LabelEncoder"""
        try:
            encoder = joblib.load(filepath)
            if not hasattr(encoder, 'classes_'):
                raise ValueError("Loaded encoder is missing classes_ attribute")
            return encoder
        except Exception as e:
            st.error(f"Encoder loading failed: {str(e)}")
            return None

    @staticmethod
    def load_model(filepath):
        """Load the prediction model"""
        try:
            model = joblib.load(filepath)
            if not hasattr(model, 'predict_proba'):
                raise ValueError("Model must have predict_proba method")
            return model
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None

# Title and description
st.title("🩺 Pancreatic Cancer Survival Predictor App")
st.markdown("""
This tool predicts survival outcomes based on tumor molecular characteristics 
and demographic factors using a machine learning model.
""")

# Load models
with st.spinner('Loading predictive models...'):
    scaler = ModelLoader.load_scaler('scaler.pkl')
    encoder = ModelLoader.load_encoder('target_encoder.pkl')
    model = ModelLoader.load_model('pancreatic_cancer_survival_model.pkl')

# Check if models loaded successfully
if None in [scaler, encoder, model]:
    st.error("Critical error: Failed to load required models. Please contact support.")
    st.stop()

# Get feature names (assuming original features are in scaler)
original_features = list(scaler.feature_names_in_)

# Input form
with st.form("patient_form"):
    st.header("Patient Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_age = st.number_input("Current Age (years)", min_value=20, max_value=100, value=65)
        tumor_purity = st.slider("Tumor Purity", min_value=0, max_value=100, value=40, step=10)
        overall_survival = st.number_input("Overall Survival (Months)", min_value=0, max_value=1000, value=100)
        stage = st.selectbox("Stage (Highest Recorded)", options=["I", "II", "III", "IV"])
    
    with col2:
        tmb = st.number_input("TMB (nonsynonymous)", min_value=0, max_value=100, value=5)
        frac_genome_altered = st.slider("Fraction Genome Altered", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        mutation_count = st.number_input("Mutation Count", min_value=0, max_value=1000, value=50)
    
    submitted = st.form_submit_button("Predict Survival Outcome")

# Prediction logic
if submitted:
    try:        
        # Create DataFrame with all features including stage
        all_features = pd.DataFrame({
            'Current Age': [current_age],
            'Tumor Purity': [tumor_purity],
            'Overall Survival (Months)': [overall_survival],
            'TMB (nonsynonymous)': [tmb],
            'Fraction Genome Altered': [frac_genome_altered],
            'Mutation Count': [mutation_count],
            'Stage (Highest Recorded)': [stage]
        })
        
        # Separate features for scaling (exclude stage)
        features_to_scale = all_features[original_features]
        scaled_features = scaler.transform(features_to_scale)
        
        # Combine scaled features with stage for prediction
        stage_encoded = pd.DataFrame({
            'Stage (Highest Recorded)': [{"I": 1, "II": 2, "III": 3, "IV": 4}[stage]]
        })
        
        # Combine all features for prediction
        final_features = np.hstack([scaled_features, stage_encoded])
        
        # Get predicted probabilities
        risk_score = model.predict_proba(final_features)[:, 1][0]
        
        # Risk stratification
        fixed_threshold = 0.4045
        risk_label = 'High Risk' if risk_score >= fixed_threshold else 'Low Risk'
        predicted_outcome = encoder.classes_[1] if risk_score >= fixed_threshold else encoder.classes_[0]
        
        # Display results
        st.subheader("📊 Prediction Results")
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Risk Score", f"{float(risk_score):.1%}")
        with cols[1]:
            st.metric("Risk Level", risk_label)
        
        # Progress bar 
        st.progress(float(risk_score))
        st.caption(f"Threshold for high risk: {fixed_threshold:.1%}")

        # Clinical interpretation
        st.subheader("💡 Clinical Interpretation")
        
        if risk_label == 'Low Risk':
            st.success(f"""
            **Recommended Actions (Low Risk - Score < {fixed_threshold:.1%}):**
            - Standard therapeutic protocol
            - Annual molecular profiling
            - Routine imaging follow-up
            - Lifestyle counseling
            """)
        else:
            st.error(f"""
            **Recommended Actions (High Risk - Score ≥ {fixed_threshold:.1%}):**
            - Immediate oncology consultation
            - Molecular tumor board review
            - Consider clinical trials
            - 3-month monitoring intervals
            - Early supportive care
            """)    
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Model information in sidebar
with st.sidebar:
    st.header("Model Information")
    st.markdown(f"""
    **Model Characteristics:**
    - Prediction threshold: 40.45%
    - Features used: {len(original_features)} variables
    - Includes demographic and molecular factors
    """)
    
    st.markdown("---")
    st.markdown("""
    **Key Features:**
    - Current Age
    - Overall Survival (Months)
    - Tumor Purity
    - Stage
    - TMB (Tumor Mutational Burden)
    - Fraction Genome Altered
    - Mutation Count
    """)

# Footer
st.markdown("---")
st.caption("""
**Disclaimer:** This tool provides statistical predictions only. 
Clinical decisions should consider the full patient context. 
Always consult oncology guidelines for treatment decisions.
""")