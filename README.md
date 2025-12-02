# 🫀 Heart Disease Prediction Using Machine Learning  

## 📊 Dataset Information  
- **Source:** [UCI Machine Learning Repository – Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)  
- **Description:** This dataset contains clinical and demographic patient data to predict the presence of heart disease. It uses a subset of **14 key features** widely adopted in ML experiments. The target variable is **binary**: 0 (no heart disease) or 1 (presence of heart disease).  
- **Cleaning:** Removed 7 entries with invalid values (`ca=4` and `thal=0`) to ensure data integrity.  

---

## 🎯 Problem Statement  
Heart disease prediction is challenging due to complex physiological relationships and data variability. This project aims to predict the presence of heart disease from patient data, identify the most influential risk factors, and provide a **robust, deployment-ready ML pipeline** for clinical decision support.  

---

## 📝 Objectives  
- Analyze the Cleveland Heart Disease dataset through **EDA** to uncover key medical patterns.  
- Preprocess and clean data, including **outlier handling**, **encoding**, and **scaling**.  
- Develop and evaluate **multiple ML models** for binary classification.  
- Identify **key predictive features** contributing to heart disease.  
- Build a **reusable pipeline** for preprocessing, modeling, and deployment.  
- Deploy the final model using **Streamlit** for interactive inference.  

---

## ⚙️ Methods and Approach  

### 🔹 Data Preparation & Feature Engineering  
- Handled missing and invalid values.  
- Capped numerical outliers using a custom transformer.  
- Applied **OneHotEncoding** for categorical variables and **StandardScaler** for numerical features.  
- Built a **preprocessing pipeline** for reproducibility and deployment.  

### 🔹 Exploratory Data Analysis (EDA)  
Key findings:  
- **Exercise-Induced Angina (`exang`)** and **ST Depression (`oldpeak`)** were the strongest predictors.  
- **Max Heart Rate (`thalach`)** and **Chest Pain Type (`cp`)** also strongly correlated with heart disease.  
- Traditional risk factors like age, cholesterol, and resting BP showed smaller differences in this dataset.  

### 🔹 Modeling  
- Compared **Random Forest**, **Logistic Regression**, and **XGBoost** in baseline and tuned forms.  
- Evaluated models using **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **ROC AUC**.  
- Hyperparameter tuning was found to be less effective due to small dataset size; baseline models performed best.  

---

## 📈 Key Results  

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|---------|-----------|--------|----------|---------|
| **Random Forest (Baseline)** | **0.800** | 0.826 | 0.792 | **0.809** | 0.849 |
| Logistic Regression (Baseline) | 0.778 | 0.818 | 0.750 | 0.783 | **0.873** |
| XGBoost (Baseline) | 0.800 | 0.826 | 0.792 | 0.809 | 0.819 |

- **Baseline Random Forest** provides the best **balanced classification** (F1 Score 0.809).  
- **Baseline Logistic Regression** offers the best **class separation** (ROC AUC 0.873).  
- Hyperparameter tuning of ensemble models led to overfitting due to the small dataset size.  

---

## 🚀 Recommendations for Future Work  
- **Expand Dataset:** Acquire more diverse patient data to improve generalization.  
- **Interpretability Tools:** Use SHAP to explain predictions at both global and local levels.  
- **Feature Engineering:** Explore additional clinical indicators and interactions.  
- **Ensemble Stacking:** Combine models to improve robustness.  

---

## 🏁 Conclusion  
This project demonstrates that **robust heart disease prediction is achievable even with small datasets**. Key clinical indicators such as **exercise-induced angina**, **ST depression**, **max heart rate**, and **chest pain type** strongly influence risk. Baseline models, particularly Random Forest and Logistic Regression, provided stable, reliable predictions suitable for deployment.  

- **Recommended Model for Deployment:** Baseline Random Forest (balanced classification).  
- **Deployment Ready:** The preprocessing pipeline and trained model can be used in a **Streamlit app** for interactive inference.  

---

Demo App: *https://nnejere-ai-ml-inference-hub-app-298cc0.streamlit.app/*

