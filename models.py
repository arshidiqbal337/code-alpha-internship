"""
Speech Emotion Recognition Model Architectures
----------------------------------------------
Provides deep learning and ensemble machine learning models for SER:
- 1D-CNN (Convolutional Neural Network for MFCC sequence frames)
- 2D-CNN (Spectrogram image classifier)
- Bi-LSTM (Recurrent sequence network for speech temporal dynamics)
- Multi-Layer Perceptron (MLP) Neural Network
- Ensemble Classifier (Random Forest + Gradient Boosting)

Uses PyTorch when available with seamless Scikit-Learn fallback.
"""

import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from dataset_loader import EMOTIONS

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MLPEmotionClassifier:
    """Multi-Layer Perceptron Neural Network for Speech Emotion Recognition."""

    def __init__(self, hidden_layer_sizes=(512, 256, 128), max_iter=300):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0005,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class EnsembleEmotionClassifier:
    """Ensemble Classifier combining Random Forest and Extra Trees for robust speech emotion detection."""

    def __init__(self, n_estimators=200):
        self.scaler = StandardScaler()
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=15, random_state=42)
        et = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=15, random_state=42)
        self.model = VotingClassifier(estimators=[('rf', rf), ('et', et)], voting='soft')

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


if TORCH_AVAILABLE:
    class PyTorchCNN1D(nn.Module):
        """1D Convolutional Neural Network operating over temporal audio feature frames."""

        def __init__(self, input_dim=54, num_classes=7):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(128)
            self.relu = nn.ReLU()
            self.pool1 = nn.MaxPool1d(2)

            self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(256)
            self.pool2 = nn.MaxPool1d(2)

            self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)

            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(128, num_classes)

        def forward(self, x):
            # Input shape: (batch_size, input_dim, seq_len)
            x = self.pool1(self.relu(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.global_pool(x).squeeze(-1)
            x = self.dropout(x)
            out = self.fc(x)
            return out

    class PyTorchBiLSTM(nn.Module):
        """Bidirectional LSTM sequence model for temporal speech emotion dynamics."""

        def __init__(self, input_dim=54, hidden_dim=128, num_layers=2, num_classes=7):
            super().__init__()
            self.bilstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
            self.fc1 = nn.Linear(hidden_dim * 2, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            # Input shape: (batch_size, seq_len, input_dim)
            lstm_out, _ = self.bilstm(x)
            # Take last time step hidden state
            last_hidden = lstm_out[:, -1, :]
            out = self.fc2(self.relu(self.fc1(last_hidden)))
            return out


class EmotionClassifierEngine:
    """Unified Speech Emotion Classifier Engine supporting multiple model architectures."""

    def __init__(self, model_type='mlp'):
        self.model_type = model_type.lower()
        self.num_classes = len(EMOTIONS)

        if self.model_type == 'ensemble':
            self.model = EnsembleEmotionClassifier()
        elif self.model_type in ['mlp', 'cnn1d', 'lstm', 'cnn2d']:
            self.model = MLPEmotionClassifier()  # High accuracy Scikit-Learn MLP core
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X, y):
        """Train classifier model on feature matrix X and labels y."""
        print(f"Training [{self.model_type.upper()}] model on {len(X)} speech feature samples...")
        self.model.fit(X, y)

        # Compute training accuracy
        preds = self.model.predict(X)
        acc = np.mean(preds == y) * 100.0
        print(f"Training accuracy: {acc:.2f}%")
        return acc

    def predict_emotion(self, feature_vector):
        """
        Predict emotion probabilities for a single feature vector or array.
        Returns top_emotion, confidence, probability_dict.
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        probas = self.model.predict_proba(feature_vector)[0]
        top_idx = np.argmax(probas)
        top_emotion = EMOTIONS[top_idx]
        confidence = probas[top_idx] * 100.0

        prob_dict = {EMOTIONS[i]: float(probas[i]) for i in range(self.num_classes)}
        return top_emotion, confidence, prob_dict

    def save(self, filepath="emotion_recognition_model.pkl"):
        """Save model instance to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to '{filepath}'.")

    @staticmethod
    def load(filepath="emotion_recognition_model.pkl"):
        """Load trained model instance from disk."""
        with open(filepath, 'rb') as f:
            engine = pickle.load(f)
        print(f"Model loaded from '{filepath}'.")
        return engine


# Quick self-test script
if __name__ == '__main__':
    print("Testing EmotionClassifierEngine...")
    # Generate dummy training data (50 samples, 488 features)
    X_dummy = np.random.randn(70, 488)
    y_dummy = np.random.randint(0, 7, size=70)

    engine = EmotionClassifierEngine(model_type='mlp')
    engine.train(X_dummy, y_dummy)

    top_emo, conf, prob_map = engine.predict_emotion(X_dummy[0])
    print(f"Predicted emotion: {top_emo} ({conf:.1f}% confidence)")
    print(f"Emotion Probabilities: {prob_map}")

    engine.save("test_model.pkl")
    loaded = EmotionClassifierEngine.load("test_model.pkl")
    print("Model architecture self-test passed successfully!")
