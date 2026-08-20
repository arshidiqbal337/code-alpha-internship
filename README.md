# Credit Scoring Model in Python 💳

A classification-based machine learning solution for evaluating financial creditworthiness, predicting credit default risk, and generating continuous credit scores (300 - 850 scale).

---

## 🎯 Objective
To evaluate an applicant's credit risk and creditworthiness using historical financial data (income, debt, credit limits, payment history, delinquencies, and inquiries).

---

## 🚀 Key Features

1. **Synthetic Data Generator** (`generate_dataset.py`):
   - Generates realistic financial profiles with demographic, liability, and payment features.
2. **Feature Engineering**:
   - **Debt-to-Income Ratio (DTI)**: `Debt / Income`
   - **Credit Utilization Ratio**: `Debt / Credit Limit`
   - **Savings-to-Debt Ratio**: `Savings / Debt`
3. **Machine Learning Classifiers**:
   - **Logistic Regression**: Linear baseline with feature standardization.
   - **Decision Tree Classifier**: Interpretable tree model for non-linear splits.
   - **Random Forest Classifier**: High-performance ensemble classifier minimizing overfitting.
4. **Comprehensive Evaluation Metrics**:
   - **Accuracy**: Overall classification accuracy on holdout test set.
   - **Precision**: Measure of positive default prediction reliability.
   - **Recall (Sensitivity)**: Ability to capture actual high-risk applicants.
   - **F1-Score**: Harmonic mean of Precision & Recall.
   - **ROC-AUC**: Receiver Operating Characteristic Area Under Curve (Discrimination metric).
5. **Scorecard Mapping**:
   - Translates predicted default probability $P(\text{Default})$ into standard 300 to 850 Credit Score range:
     $$\text{Credit Score} = 850 - \left(P(\text{Default}) \times 550\right)$$
6. **Interactive Dashboard**:
   - Beautiful dashboard (`dashboard.html`) visualizing performance metrics, ROC curves, confusion matrices, and a live simulator.

---

## 📊 Model Performance Summary

| Algorithm | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9610** | **91.0%** | **78.1%** | **55.6%** | **0.649** |
| **Logistic Regression** | 0.9509 | 91.0% | 76.5% | 57.8% | 0.658 |
| **Decision Tree** | 0.8715 | 91.3% | 73.2% | 66.7% | 0.698 |

---

## 📁 Repository Structure

```
project 1/
│
├── credit_scoring_model.py   # Main pipeline (feature engineering, training, evaluation, scoring)
├── generate_dataset.py       # Synthetic credit dataset generator
├── interactive_scorer.py     # Interactive CLI tool to test custom financial profiles
├── dashboard.html            # Web dashboard with visual charts and score simulator
├── roc_curves.png            # Model ROC curves visual chart
├── confusion_matrices.png    # Model confusion matrices heatmap
└── README.md                 # Complete documentation
```

---

## 🛠️ Usage Instructions

### 1. Run the Full Model Pipeline
To generate the dataset, train all classification models, compute metrics, and export performance charts:
```bash
python credit_scoring_model.py
```

### 2. Run Interactive Applicant Assessor
To test individual credit applicants interactively via the CLI:
```bash
python interactive_scorer.py
```

### 3. Open the Interactive Visual Dashboard
Open `dashboard.html` in any web browser to view performance charts and test the live score calculator!
