"""
train_evaluate.py
-----------------
Train and evaluate 4 Classification Machine Learning Algorithms:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost

Evaluates performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Saves trained models to the 'models/' folder.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_loader import get_preprocessed_dataset


def get_models():
    """
    Returns a dictionary of the 4 requested classification algorithms.
    """
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42) if HAS_XGBOOST else GradientBoostingClassifier(random_state=42)
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb_model
    }


def evaluate_model(model, X_test_scaled, y_test):
    """
    Computes key performance metrics for a trained classification model.
    """
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4)
    }


def train_and_evaluate_dataset(dataset_name, output_dir="models"):
    """
    Trains all 4 algorithms on a given dataset and saves models.
    """
    print(f"\n==========================================")
    print(f"   Training Models on: {dataset_name.upper()}")
    print(f"==========================================")

    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, X_raw = get_preprocessed_dataset(dataset_name)

    # Save scaler & feature names for inference
    joblib.dump(scaler, os.path.join(output_dir, f"{dataset_name}_scaler.pkl"))
    joblib.dump(feature_names, os.path.join(output_dir, f"{dataset_name}_features.pkl"))

    models = get_models()
    results = []

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate performance
        metrics = evaluate_model(model, X_test_scaled, y_test)
        metrics["Algorithm"] = name
        metrics["Dataset"] = dataset_name
        results.append(metrics)

        # Save trained model to disk
        model_filename = os.path.join(output_dir, f"{dataset_name}_{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_filename)
        print(f" [OK] {name:<20} | Accuracy: {metrics['Accuracy']:.4f} | F1-Score: {metrics['F1-Score']:.4f}")

    results_df = pd.DataFrame(results)[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    return results_df


def main():
    datasets = ["breast_cancer", "diabetes", "heart_disease"]
    all_results = {}

    for ds in datasets:
        df_res = train_and_evaluate_dataset(ds)
        all_results[ds] = df_res
        
        # Save comparison results CSV
        df_res.to_csv(os.path.join("models", f"{ds}_results.csv"), index=False)

    print("\n==========================================")
    print("        SUMMARY RESULTS ACROSS ALL DATASETS")
    print("==========================================")
    for ds, df_res in all_results.items():
        print(f"\n--- Dataset: {ds.upper()} ---")
        print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
