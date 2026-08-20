import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

from generate_dataset import generate_credit_dataset

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11

def perform_feature_engineering(df):
    """
    Creates intuitive financial metrics from raw financial history.
    1. Debt-to-Income Ratio (DTI)
    2. Credit Utilization Ratio
    3. Savings-to-Debt Ratio
    """
    data = df.copy()
    
    # Avoid division by zero with small epsilon (+ 1.0)
    data['DTI_Ratio'] = data['Debt'] / (data['Income'] + 1.0)
    data['Credit_Utilization'] = data['Debt'] / (data['Credit_Limit'] + 1.0)
    data['Savings_to_Debt'] = data['Savings_Balance'] / (data['Debt'] + 1.0)
    
    return data

def train_and_evaluate(df):
    """
    Trains Classification Models (Logistic Regression, Decision Tree, Random Forest)
    and evaluates using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    # 1. Feature Engineering
    engineered_df = perform_feature_engineering(df)
    
    # 2. Separate Features (X) and Target (y)
    X = engineered_df.drop(columns=['Credit_Risk'])
    y = engineered_df['Credit_Risk']
    
    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Feature Scaling (Standardization for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Define Classification Algorithms
    models = {
        'Logistic Regression': (LogisticRegression(random_state=42), X_train_scaled, X_test_scaled),
        'Decision Tree': (DecisionTreeClassifier(max_depth=5, random_state=42), X_train, X_test),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), X_train, X_test)
    }
    
    # 6. Evaluation Loop
    results = []
    trained_models = {}
    
    plt.figure(figsize=(10, 7))
    
    for name, (model, train_data, test_data) in models.items():
        # Train model
        model.fit(train_data, y_train)
        trained_models[name] = model
        
        # Predict classes and probabilities
        y_pred = model.predict(test_data)
        y_prob = model.predict_proba(test_data)[:, 1]
        
        # Calculate key performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    
    # Complete ROC Curve Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves - Credit Risk Classification Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    plt.close()
    
    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (name, (model, _, test_data)) in enumerate(models.items()):
        y_pred = model.predict(test_data)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f"{name}\nConfusion Matrix")
        axes[idx].set_xlabel('Predicted Label (0=Good, 1=Bad)')
        axes[idx].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    plt.close()
    
    results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
    
    return results_df, trained_models, scaler, list(X.columns)

def calculate_credit_score(default_prob):
    """
    Translates default probability into a traditional Credit Score (300 to 850 scale).
    - 0% default prob -> 850 credit score (Excellent)
    - 100% default prob -> 300 credit score (Poor)
    """
    score = 850 - int(default_prob * 550)
    return max(300, min(850, score))

def score_applicant(applicant_data, model, scaler, feature_names, is_scaled=False):
    """
    Predicts creditworthiness and credit score for a single applicant.
    """
    # 1. Convert applicant dictionary to DataFrame
    df_single = pd.DataFrame([applicant_data])
    
    # 2. Perform Feature Engineering
    engineered_single = perform_feature_engineering(df_single)
    engineered_single = engineered_single[feature_names]
    
    # 3. Scale if required by model
    if is_scaled:
        input_data = scaler.transform(engineered_single)
    else:
        input_data = engineered_single
        
    # 4. Predict Risk & Probability
    prob_default = model.predict_proba(input_data)[0, 1]
    is_high_risk = prob_default > 0.40
    credit_score = calculate_credit_score(prob_default)
    
    # 5. Rating Category
    if credit_score >= 750:
        rating = "Excellent (Low Risk)"
    elif credit_score >= 670:
        rating = "Good (Moderate Risk)"
    elif credit_score >= 580:
        rating = "Fair (High Risk)"
    else:
        rating = "Poor (Very High Risk)"
        
    return {
        'Credit Score': credit_score,
        'Default Probability (%)': round(prob_default * 100, 2),
        'Risk Status': 'High Risk / Reject' if is_high_risk else 'Low Risk / Approve',
        'Credit Rating': rating
    }

if __name__ == '__main__':
    print("=" * 60)
    print("      CREDIT SCORING MODEL - SEAMLESS PIPELINE")
    print("=" * 60)
    
    # Step 1: Load or Generate Dataset
    print("\n[Step 1] Loading synthetic financial dataset...")
    data = generate_credit_dataset(num_samples=1500)
    
    # Step 2: Train Models & Evaluate
    print("[Step 2] Engineering features & training classification models...")
    metrics_summary, trained_models, scaler, feature_names = train_and_evaluate(data)
    
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(metrics_summary.to_string(index=False))
    
    # Step 3: Test Applicant Evaluation
    best_model_name = metrics_summary.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    is_scaled = (best_model_name == 'Logistic Regression')
    
    print(f"\n[Step 3] Evaluating sample applicants using best model ({best_model_name})...")
    
    # Sample Applicant 1: Strong Financial Profile
    applicant_1 = {
        'Age': 38,
        'Income': 95000.0,
        'Debt': 8500.0,
        'Credit_Limit': 40000.0,
        'Savings_Balance': 32000.0,
        'Delinquencies': 0,
        'Credit_Inquiries': 1,
        'Credit_Age_Years': 12,
        'Num_Open_Accounts': 6
    }
    
    # Sample Applicant 2: Vulnerable Financial Profile
    applicant_2 = {
        'Age': 26,
        'Income': 28000.0,
        'Debt': 22000.0,
        'Credit_Limit': 15000.0,
        'Savings_Balance': 1200.0,
        'Delinquencies': 3,
        'Credit_Inquiries': 4,
        'Credit_Age_Years': 2,
        'Num_Open_Accounts': 8
    }
    
    result_1 = score_applicant(applicant_1, best_model, scaler, feature_names, is_scaled)
    result_2 = score_applicant(applicant_2, best_model, scaler, feature_names, is_scaled)
    
    print("\n--- Applicant 1 (High Income, Low Delinquency) ---")
    for key, val in result_1.items():
        print(f"  {key}: {val}")
        
    print("\n--- Applicant 2 (High Debt, Multiple Delinquencies) ---")
    for key, val in result_2.items():
        print(f"  {key}: {val}")

    print("\n[Complete] Performance charts saved to 'roc_curves.png' and 'confusion_matrices.png'.")
