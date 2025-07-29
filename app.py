import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="Pancreatic Cancer Survival Prediction", layout="centered")
st.markdown("## Pancreatic Cancer Risk Identification and Treatment Recommendation")
st.markdown("""Predict a patient’s risk of poor survival using clinical and genomic features this helps doctors identify high-risk individuals for timely intervention and more personalized care.""")


# ----------------------------
# Load model and preprocessors
# ----------------------------

@st.cache_resource
def load_model():
    try:
        model = joblib.load("pancreatic_cancer_survival_model.pkl")
        assert hasattr(model, "predict_proba"), "Model lacks predict_proba"
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def load_preprocessors():
    try:
        imputer = joblib.load("imputer.pkl")
        scaler = joblib.load("scaler.pkl")
        encoder = joblib.load("target_encoder.pkl")
        return imputer, scaler, encoder
    except Exception as e:
        st.error(f"Failed to load preprocessors: {e}")
        return None, None, None

# Load all at once
model = load_model()
imputer, scaler, encoder = load_preprocessors()

# ----------------------------
# App Layout
# ----------------------------


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


# ----------------------------
# Process Prediction
# ----------------------------

scaler_columns = ['Current Age', 'Tumor Purity', 'Overall Survival (Months)', 'TMB (nonsynonymous)', 'Mutation Count', 'Fraction Genome Altered']

model_feature_order = ['Current Age', 'Tumor Purity', 'Overall Survival (Months)',
       'TMB (nonsynonymous)', 'Stage (Highest Recorded)', 'Mutation Count',
       'Fraction Genome Altered']

original_features = model_feature_order

if submitted:
    try:
        # Collect input into a DataFrame
        input_df = pd.DataFrame({
            'Current Age': [current_age],
            'Tumor Purity': [tumor_purity],
            'Overall Survival (Months)': [overall_survival],
            'TMB (nonsynonymous)': [tmb],
            'Fraction Genome Altered': [frac_genome_altered],
            'Mutation Count': [mutation_count],
            'Stage (Highest Recorded)': [stage]
        })

        # Ensure correct column order
        input_df = input_df[model_feature_order]  # <-- make sure this matches your training set!

        # Scale numerical columns only
        input_scaled = input_df.copy()
        input_scaled[scaler_columns] = scaler.transform(input_df[scaler_columns])


        stage_mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
        input_scaled["Stage (Highest Recorded)"] = input_scaled["Stage (Highest Recorded)"].map(stage_mapping)


        # Convert to standard float type
        input_scaled = input_scaled.astype(float)

        # Predict risk score
        risk_score = model.predict_proba(input_scaled)[:, 1][0]
        fixed_threshold = 0.4045
        risk_label = "High Risk" if risk_score >= fixed_threshold else "Low Risk"

        # Display results
        st.subheader("📊 Prediction Results")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Risk Score", f"{risk_score:.1%}")
        with cols[1]:
            st.metric("Risk Level", risk_label)
        with cols[2]:
            st.metric("Threshold", f"{fixed_threshold:.1%}")

        st.progress(float(risk_score))

        st.subheader("💡 Clinical Interpretation")
        if risk_label == "Low Risk":
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

    st.markdown("""
    <p style='text-align: center; font-size: 12px;'>
    Built by The Insight Lab
    </p>
    """, unsafe_allow_html=True)