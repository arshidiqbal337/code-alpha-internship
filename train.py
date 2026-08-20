"""
Speech Emotion Recognition Model Training and Evaluation Script
--------------------------------------------------------------
Trains deep learning and ensemble ML models on speech dataset features.
Outputs accuracy, precision, recall, F1-scores, and generates visual confusion matrices.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset_loader import DatasetLoader, EMOTIONS
from models import EmotionClassifierEngine


def train_and_evaluate(data_path="sample_data", model_type="mlp", model_output="emotion_recognition_model.pkl"):
    """Trains emotion recognition model and evaluates performance metrics."""
    print("=" * 65)
    print("      SPEECH EMOTION RECOGNITION (SER) - MODEL TRAINING      ")
    print("=" * 65)

    loader = DatasetLoader(data_path=data_path)
    X, y, file_paths = loader.load_dataset_features(feature_type='aggregated')

    if len(X) == 0:
        raise ValueError("No feature data extracted. Please check dataset path.")

    print(f"\nDataset loaded successfully: {len(X)} audio samples across {len(EMOTIONS)} emotions.")
    print(f"Feature Vector Dimension: {X.shape[1]}")

    # Train / Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")

    # Initialize and train model
    engine = EmotionClassifierEngine(model_type=model_type)
    engine.train(X_train, y_train)

    # Evaluate on Test Set
    test_preds = engine.model.predict(X_test)
    acc = accuracy_score(y_test, test_preds) * 100.0

    print("\n" + "=" * 55)
    print(f"  MODEL EVALUATION RESULTS [{model_type.upper()}]")
    print("=" * 55)
    print(f"  --> Test Accuracy: {acc:.2f}%\n")

    report = classification_report(y_test, test_preds, target_names=EMOTIONS, zero_division=0)
    print(report)

    # Save model artifact
    engine.save(model_output)

    # Generate and save Confusion Matrix Plot
    cm = confusion_matrix(y_test, test_preds)
    plot_confusion_matrix(cm, model_type=model_type, acc=acc)

    return engine, acc


def plot_confusion_matrix(cm, model_type="MLP", acc=0.0):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTIONS, yticklabels=EMOTIONS
    )
    plt.title(f"Speech Emotion Recognition Confusion Matrix ({model_type.upper()} - {acc:.1f}% Acc)")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to '{plot_path}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition Model")
    parser.add_argument("--data_path", type=str, default="sample_data", help="Path to audio dataset directory")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "ensemble"], help="Model architecture")
    parser.add_argument("--output", type=str, default="emotion_recognition_model.pkl", help="Output model path")

    args = parser.parse_args()
    train_and_evaluate(data_path=args.data_path, model_type=args.model_type, model_output=args.output)
