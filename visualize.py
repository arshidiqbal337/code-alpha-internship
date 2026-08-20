"""
visualize.py
------------
Creates insightful visual evaluation graphics:
1. Algorithm Performance Comparisons (Accuracy & F1-Score Bar Chart)
2. Confusion Matrix Heatmaps
3. Feature Importance Plots (Random Forest / XGBoost)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_loader import get_preprocessed_dataset

# Set attractive aesthetic styling
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", palette="muted")


def plot_model_comparison(dataset_name, results_csv, output_dir="plots"):
    """
    Plots a bar chart comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC
    across the 4 algorithms for a dataset.
    """
    if not os.path.exists(results_csv):
        print(f"Results CSV {results_csv} not found. Run train_evaluate.py first.")
        return

    df = pd.read_csv(results_csv)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    df_melted = pd.melt(df, id_vars=['Algorithm'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        var_name='Metric', value_name='Score')

    ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Algorithm')
    plt.title(f"Model Performance Comparison - {dataset_name.replace('_', ' ').title()}", fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add numeric labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{dataset_name}_model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" [OK] Saved plot: {plot_path}")


def plot_confusion_matrices(dataset_name, models_dir="models", output_dir="plots"):
    """
    Plots confusion matrices side-by-side for all 4 trained algorithms.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, X_raw = get_preprocessed_dataset(dataset_name)

    algorithms = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for idx, algo in enumerate(algorithms):
        model_path = os.path.join(models_dir, f"{dataset_name}_{algo.replace(' ', '_').lower()}.pkl")
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Disease"])
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(algo, fontsize=12, fontweight='bold')
        axes[idx].grid(False)

    plt.suptitle(f"Confusion Matrices - {dataset_name.replace('_', ' ').title()}", fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrices.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] Saved confusion matrices: {plot_path}")


def plot_feature_importance(dataset_name, models_dir="models", output_dir="plots"):
    """
    Plots Feature Importance from Random Forest / XGBoost models.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{dataset_name}_random_forest.pkl")
    features_path = os.path.join(models_dir, f"{dataset_name}_features.pkl")

    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        return

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], hue=np.array(feature_names)[indices], palette="viridis", legend=False)
        plt.title(f"Feature Importance (Random Forest) - {dataset_name.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        plt.xlabel("Relative Importance Score", fontsize=11)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{dataset_name}_feature_importance.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f" [OK] Saved feature importance plot: {plot_path}")


def main():
    datasets = ["breast_cancer", "diabetes", "heart_disease"]

    print("\n==========================================")
    print("      GENERATING VISUAL PLOTS & CHARTS")
    print("==========================================")

    for ds in datasets:
        results_csv = os.path.join("models", f"{ds}_results.csv")
        plot_model_comparison(ds, results_csv)
        plot_confusion_matrices(ds)
        plot_feature_importance(ds)


if __name__ == "__main__":
    main()
