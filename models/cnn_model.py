"""
Convolutional Neural Network (CNN) architecture for isolated character recognition.
Optimized for 28x28 grayscale images from MNIST and EMNIST datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class CharacterCNN(nn.Module):
    """
    Deep Convolutional Neural Network for handwritten character & digit recognition.
    
    Architecture:
      - Conv Block 1: Conv2D(1->32) + BN + ReLU + Conv2D(32->64) + BN + ReLU + MaxPool(2x2) + Dropout(0.25)
      - Conv Block 2: Conv2D(64->128) + BN + ReLU + Conv2D(128->128) + BN + ReLU + MaxPool(2x2) + Dropout(0.25)
      - Classifier: FC(128*7*7 -> 256) + BN + ReLU + Dropout(0.5) + FC(256 -> num_classes)
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES_EMNIST):
        super(CharacterCNN, self).__init__()
        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.dropout1 = nn.Dropout(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        self.dropout2 = nn.Dropout(0.25)

        # Dense Classifier
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten & Dense
        x = x.view(x.size(0), -1)  # (batch_size, 128 * 7 * 7)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        logits = self.fc2(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns class probability distribution using Softmax.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
