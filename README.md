# Heart-Disease-Prediction-with-Machine-Learning

# ❤️ Heart Disease Prediction Using Machine Learning

This project develops a complete machine learning pipeline for predicting the likelihood of **heart disease** using clinical and demographic patient data. The project emphasizes **data preprocessing**, **exploratory data analysis (EDA)**, **model training**, **evaluation**, and **deployment-ready inference**.

---

## 💡 Introduction

The human heart pumps blood throughout the body, sustaining life. When its structure or function is compromised, cardiovascular diseases (CVDs) arise, including coronary artery disease, heart failure, and arrhythmias.

According to the World Health Organization (WHO, 2023), **CVDs are the leading cause of death globally**, responsible for 17.9 million deaths annually. Early detection is key, yet traditional diagnosis can be slow or subjective.

Machine learning offers a data-driven approach to identify subtle patterns in medical data, improving risk prediction and decision support. This project applies ML techniques to predict the likelihood of heart disease using structured patient health indicators.

---

## 📌 Project Overview & Problem Statement

### Project Overview
This project explores how machine learning can predict the presence of heart disease based on clinical data such as age, cholesterol, chest pain type, blood pressure, and heart rate. The goal is to build a reliable model that not only predicts but also highlights which health factors contribute most to heart disease risk, aiding prevention and medical insight.


[Image of machine learning workflow diagram showing steps from data collection to model deployment]


### Problem Statement
Heart disease prediction remains difficult due to complex physiological relationships and data variability. This project seeks to answer the question: **Can patient health data be used to accurately predict heart disease using machine learning?** By addressing this, the project aims to enhance diagnostic efficiency and uncover the most influential predictors.

---

## 🎯 Objectives

1.  **Analyze** the heart disease dataset to uncover relationships among key medical variables through comprehensive EDA.
2.  **Preprocess and clean** the data, including handling outliers and encoding, to ensure high-quality input for model training.
3.  **Develop and evaluate** multiple machine learning models for **binary classification** (disease vs. no disease).
4.  **Identify** the most influential features contributing to heart disease prediction.
5.  **Present** insights that can guide data-driven healthcare decision-making and early screening.
6.  **Deploy** the final model and preprocessing pipeline using **Streamlit** for interactive inference.

---

## 🧠 Dataset Description & Cleaning

This project uses the **Cleveland Heart Disease Dataset** from the **UCI Machine Learning Repository**.

The database uses a subset of **14 key features** for common ML experiments. The target variable, originally integer-valued from 0 (no disease) to 4 (severe disease), is treated as a **binary classification problem** for this project:

* **Value 0:** No significant heart disease ($< 50\%$ diameter narrowing).
* **Value 1:** Presence of heart disease ($\ge 50\%$ diameter narrowing).

### Data Cleaning Action
Based on published notes from the dataset, entries containing invalid values (e.g., `ca=4` and `thal=0`) were treated as missing and removed. **7 faulty data entries were dropped** to ensure data integrity.

### Key Features Used

| Column | Description |
| :--- | :--- |
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic) |
| `trestbps` | Resting blood pressure on admission to the hospital (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (**0**: normal, **1**: ST-T wave abnormality, **2**: left ventricular hypertrophy) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping) |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia type (1 = fixed defect, 2 = normal, 3 = reversible defect) |
| `target` | Diagnosis of heart disease (**0** = no disease, **1** = disease) |

---

### 🛑 Baseline Vs. Tuned Models Summary

The consistent outperformance of baseline models, particularly the ensemble classifiers **(Random Forest and XGBoost)**, confirms that aggressive hyperparameter optimization was counterproductive for this project.

1.  **Small Dataset Overfitting:** The primary cause is **overfitting** to the cross-validation (CV) folds within the limited training data ($N \approx 250$). The grid search found parameter combinations that were highly specific to the noise of the small folds, failing to generalize to the unseen test set.
2.  **Robust Default Settings:** The default parameters in scikit-learn are often designed to provide a good level of regularization and generalize well across diverse datasets. For Random Forest, the default settings created enough ensemble diversity (low variance) to capture patterns effectively.
3.  **Data Simplicity:** The fact that the Baseline Logistic Regression achieved the best ROC AUC (0.873) suggests the key relationships between the preprocessed features and the target are predominantly **linear or simple**, rendering the complexity introduced by tuning deep, non-linear ensemble models unnecessary and harmful.

---
## 🧪 Exploratory Data Analysis (EDA) Insights

The EDA phase was critical in uncovering the strongest predictors, many of which confirm clinical expectations:

### Strongest Predictors (High Correlation)

* **Exercise-Induced Angina (`exang`):** The single strongest predictor (Negative Correlation: -0.44). Patients **without** exercise-induced angina had a significantly *higher* incidence of heart disease in this cohort, suggesting a lack of the classic symptom can be a critical sign for this disease type.
* **ST Depression (`oldpeak`):** Strong inverse relationship (Correlation: -0.43). A **lower** mean ST depression during exercise is indicative of heart disease presence.
* **Max Heart Rate (`thalach`):** Strong positive correlation (Correlation: 0.42). Patients with heart disease achieved a **significantly higher** mean maximum heart rate (158.6 bpm vs 138.9 bpm for the healthy group).
* **Chest Pain Type (`cp`):** Strong positive correlation (0.43). **Atypical Angina (type 1)** and **Non-Anginal Pain (type 2)** were highly prevalent in patients with heart disease.

### Secondary Predictors

* **Resting ECG (`restecg`):** **ST-T wave abnormality (1)** is significantly more prevalent in the heart disease group, confirming its clinical relevance.
* **ST Slope (`slope`):** A **Downsloping ST segment (type 2)** is confirmed as a risk indicator.
* **Age, Resting BP, Cholesterol:** These features showed only a small, non-discriminative difference between the healthy and disease groups in this specific dataset.
* **Fasting Blood Sugar (`fbs`):** This feature was **not a strong predictor** in this dataset.

---

## 🔧 Preprocessing & Pipeline

The preprocessing pipeline ensures consistency between training and deployment:

### 1. Outlier Handling
* A custom `cap_outliers()` transformer was used.
* It caps all numerical features at the **1st & 99th percentile** to stabilize training and mitigate the influence of extreme physiological outliers.

### 2. Encoding & Scaling
* **Encoding:** `OneHotEncoder(handle_unknown='ignore')` was applied to categorical variables.
* **Scaling:** `StandardScaler()` was applied to numerical features to standardize their distributions.

### 3. Pipeline Deployment
A complete Scikit-Learn `Pipeline` and `ColumnTransformer` were built and saved as `preprocessor.pkl`, ensuring that raw data input in the production environment receives the exact same transformations as the training data.

---

## 🚀 Model Training & Evaluation

The project compared three classification models using baseline parameters and then attempted hyperparameter tuning.

### Baseline Models Performance Summary

| Model | Accuracy | Precision | Recall | **F1 Score** | **ROC AUC** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.800** | 0.826 | 0.792 | **0.809** | 0.849 |
| Logistic Regression | 0.778 | 0.818 | 0.750 | 0.783 | **0.873** |
| XGBoost | 0.800 | 0.826 | 0.792 | 0.809 | 0.819 |

### Final Model Selection

The final modeling decision focused on the trade-off between class separation (ROC AUC) and balanced classification (F1 Score).

1.  **Best for Balanced Classification:** **Baseline Random Forest** (F1 Score: 0.809, Accuracy: 0.800). Its high F1 Score represents the best balance between False Positives and False Negatives, which is crucial for decision support.
2.  **Best for Class Discrimination:** **Baseline Logistic Regression** (ROC AUC: 0.873).

> **Crucial Finding:** For this small dataset ($N=303$), **the Baseline Models consistently outperformed their highly-tuned counterparts**. Aggressive hyperparameter tuning was found to be detrimental, leading to overfitting in the ensemble models (RF and XGBoost).

---

## 📢 Recommendations & Conclusion

### General Recommendations

* **Select Final Model:** The **Baseline Random Forest** is the recommended final deployment candidate. Its highest F1 Score (0.809) offers the best clinical trade-off.
* **Enhance Interpretability:** Utilize SHAP analysis to provide clear, local, and global explanations for the model's risk scores, enhancing physician trust and understanding.
* **Data Expansion:** The most critical project limitation is the size of the dataset. For true production readiness, efforts must focus on acquiring a larger, more diverse patient dataset to ensure model generalization.

### Conclusion Summary

The Heart Disease Prediction project successfully identified and leveraged key patient indicators, confirming that **exercise\_induced\_angina**, **st\_depression**, and **max\_heart\_rate** are the most influential clinical factors.

The project demonstrated that, despite the small dataset, robust prediction is possible: the Baseline Logistic Regression achieved superior class separation (ROC AUC: 0.873), and the **Baseline Random Forest achieved the most balanced classification (F1: 0.809)**. The success of the baseline models strongly suggests that for this specific problem size, default regularization generalized best, providing a reliable and stable prediction tool ready for the planned Streamlit deployment.
