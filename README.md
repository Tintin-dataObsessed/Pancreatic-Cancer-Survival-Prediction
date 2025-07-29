## 🧬Pancreatic Cancer Risk Identification and Treatment Recommendation Project

This project explores how machine learning models can help predict the survival outcome of pancreatic cancer patients based on a mix of clinical and genomic data. The goal is to support clinicians in identifying high-risk patients for prioritized intervention and personalized treatment strategies.

---

###  Introduction and Background
Pancreatic cancer is known for its poor prognosis and late detection. This project aims to build predictive models to classify patients into two categories: living (0) or deceased (1), using structured features such as mutation count, tumor location, and various biomarkers. The workflow includes preprocessing, model training, performance evaluation, and threshold tuning to optimize the model's clinical relevance.


### Project Objectives
- Extract and isolate pancreatic cancer cases from the broader MSK-CHORD dataset (25,000+ patients).
- Clean and preprocess clinical, pathological, and genomic features 
- Build and evaluate machine learning models to classify survival outcomes (Alive vs. Deceased).
- Interpret feature importance to identify potential prognostic biomarkers.
- Demonstrate how the model could support personalized treatment and early risk counseling.

---

### 1. Data Understanding ( Dataset: MSK-CHORD (Nature 2024) )
The dataset is sourced from the MSK-CHORD 2024 clinical-genomic database, comprising over 25,000 tumors from 24,950 patients sequenced via **MSK-IMPACT**.

- Genomic + Clinical data
- Features include age, stage, TMB, tumor purity, mutation count, sample class, and more
- Subset: 3,109 pancreatic cancer patient records
- License: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- Data Source – cBioPortal (https://www.cbioportal.org/study/summary?id=msk_chord_2024)

### 2. Exploratory Data Analysis (EDA)

Initial EDA provided insights into Age, common tumor sites, and feature distributions. 

#### Distribution of Patient Age
<img src="https://github.com/user-attachments/assets/7dce2ab6-2700-475b-98ec-545a48e4b78f" width="600"/>

#### Distribution of Patient Status
<img src="https://github.com/user-attachments/assets/1ff7b8c3-f0ce-4f99-8307-a7957399a461" width="600"/>

#### Metastatic Site and Survival Status
<img src="https://github.com/user-attachments/assets/6f5c993f-d398-4dab-8acf-964a390445dd" width="600"/>

### 3. Model Development
Three classification models were developed and evaluated:

#### 3.1 Logistic Regression

As a baseline model, Logistic Regression is simple, fast, and interpretable. Its main strength lies in the clear understanding of coefficients and their impact on predictions.

* Strengths : High recall for class 1 (0.91) suggests that the model correctly identifies most patients who are deceased.
* Weaknesses : However, it performs poorly for class 0, which may lead to overestimation of mortality risk.
* Overall: Logistic Regression provides a good starting point with an AUC-ROC of 0.7688.

#### 3.2 Random Forest Classifier

Random Forests are ensemble models that aggregate multiple decision trees to improve generalization. This model captured non-linear relationships and interactions between features better than Logistic Regression.

* Strengths : High recall (0.88) and precision (0.80) for class 1 make it strong in identifying deceased patients.
* Weaknesses : It still struggles with class 0 , meaning many living patients are misclassified as deceased.
* Overall: AUC-ROC improved to 0.7902 compared to Logistic Regression.

#### 3.3 XGBoost Classifier

XGBoost is a powerful gradient boosting framework designed to handle structured and imbalanced datasets. It generally outperforms simpler models in complex tasks.

* Strengths: Excellent recall (0.85) and F1-score (0.82) for class 1. XGBoost handles class imbalance well, with a robust AUC of 0.8027.
* Weaknesses: Precision and recall for class 0 are still limited (0.62 and 0.54 ).
* Overall: XGBoost showed the best balance between predictive performance and model consistency, even before tuning.

### 4. Cross-Validation

5-fold cross-validation was used to evaluate model generalizability. Mean ROC-AUC and standard deviation were reported:

* **Logistic Regression**: 0.7530 ± 0.0761
* **Random Forest**: 0.6647 ± 0.0647
* **XGBoost**: 0.6409 ± 0.0726

Logistic Regression had the best cross-validated ROC-AUC, but XGBoost performed better on the test set, making it more suitable for deployment.

### 5. Hyperparameter Tuning

GridSearchCV was used to fine-tune Random Forest and XGBoost models:

* **Tuned Random Forest**: Slight improvement to AUC-ROC (0.8040). Class 1 metrics remained strong, while class 0 still had recall issues (0.53).
* **Tuned XGBoost**: Achieved the highest AUC-ROC (0.8237). Recall for class 1 improved to 0.87, and F1-score rose to 0.85.

These results confirmed XGBoost as the best-performing model overall.

### 6. Threshold Tuning and Risk Stratification

Rather than using the default 0.5 threshold, the model was optimized based on F1-score. The optimal threshold was found to be **0.4332**.

* At this threshold:

  * Class 1 Recall: 0.93
  * Class 0 Precision: 0.77
  * Accuracy: 0.80

Patients were stratified as **High Risk** (score ≥ 0.4332) or **Low Risk** based on this cutoff, improving clinical interpretability.

### 7. Feature Importance

Feature importances were extracted from both Random Forest and XGBoost models. Top features (e.g., Mutation Count, Stage, TMB) aligned with known clinical indicators, supporting model credibility.

### 8. Model Deployment

The best model, tuned XGBoost , was saved for deployment.

### 9. Clinical Utility Summary

* High-risk patients can be targeted for aggressive or experimental treatments.
* Low-risk patients may be placed under routine monitoring.
* The model supports personalized care by stratifying patients based on risk scores, making it highly relevant in real-world oncology settings.

### 10. Findings and Recommendations

**Key Findings**

- The majority of patients in the dataset did not survive, highlighting the need for early, data-driven risk prediction in pancreatic cancer.
- The best-performing model (XGBoost with tuned threshold) achieved high recall (0.93) for identifying deceased patients — ideal for catching high-risk cases early.
- Tumor stage ,mutation count, and MSI score were the most important features — emphasizing the role of genomic profiling in survival prediction.
- Living patients were harder to predict accurately, suggesting missing or underrepresented recovery-related data.

**Recommendations**

- Use the model for early risk stratification at the point of diagnosis or treatment planning.
- Enhance data collection for survivors by capturing post-treatment outcomes, lifestyle variables, and co-morbidities.
- Integrate the model in clinical workflows to support ICU triage, follow-up scheduling, and palliative care prioritization.
- Use in clinical trials to select high-risk candidates more effectively, improving study design and therapeutic outcomes.
- Scale the model to regional hospitals, especially in resource-limited settings, since it only requires standard clinical/genomic data.

---

### Streamlit Application: [🌐 View the App](https://pancreatic-cancer-survival-prediction-a96rvwwkx55pmt95qwhjwa.streamlit.app/)

### Non-Technical Presentation : [📄 View the presentation](./Presentation.pdf)

---

**Collaborators:**

1. joshua karanja
2. michelle chekwoti
3. myrajoy kiganane
4. christopher katimbwa
5. robert sumaili

