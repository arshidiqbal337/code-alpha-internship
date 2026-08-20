"""
Convolutional Recurrent Neural Network (CRNN) model for sequence (word/sentence) handwritten text recognition.
Combines CNN feature extraction, Bidirectional LSTM sequence modeling, and CTC loss decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class WordCRNN(nn.Module):
    """
    CRNN Architecture:
      1. CNN Feature Extractor: extracts feature map sequence from input word images (32 x W)
      2. Map-to-Sequence: converts spatial feature maps into a sequence of feature vectors
      3. Recurrent Layer: 2-layer Bidirectional LSTM for contextual sequence modeling
      4. Linear Classifier & CTC Decoder: output class probabilities for CTC Loss computation
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES_EMNIST, hidden_size: int = 128):
        super(WordCRNN, self).__init__()
        # +1 for CTC blank token (at index 0)
        self.num_classes_ctc = num_classes + 1
        self.hidden_size = hidden_size

        # CNN Feature Extractor
        # Input shape: (B, 1, 32, W)
        self.cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (B, 64, 16, W/2)

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (B, 128, 8, W/4)

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (B, 256, 4, W/4)

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (B, 512, 2, W/4)

            # Conv Block 5
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # (B, 512, 1, W/4 - 1)
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # Bidirectional LSTM Recurrent Layers
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False  # Sequence length first for PyTorch CTC loss compatibility: (T, B, H)
        )

        # Output Transcription Layer
        self.fc = nn.Linear(hidden_size * 2, self.num_classes_ctc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, 1, height=32, width)
        Returns:
            Log probabilities of shape (seq_len, batch_size, num_classes_ctc)
        """
        features = self.cnn(x)  # Shape: (B, 512, 1, W_seq)
        features = features.squeeze(2)  # Shape: (B, 512, W_seq)

        # Map to Sequence: transpose to (W_seq, B, 512)
        seq_features = features.permute(2, 0, 1)

        # Bidirectional LSTM forward
        rnn_out, _ = self.rnn(seq_features)  # Shape: (W_seq, B, 2 * hidden_size)

        # Linear projection to classes
        logits = self.fc(rnn_out)  # Shape: (W_seq, B, num_classes_ctc)

        # Log Softmax for CTC Loss calculation
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs

    def decode_greedy(self, log_probs: torch.Tensor, char_list: list = config.CRNN_CHAR_LIST) -> list:
        """
        Greedy CTC Decoder:
        Selects argmax at each time step, collapses contiguous duplicate indices, and removes blank tokens (index 0).
        
        Args:
            log_probs: Tensor of shape (T, B, num_classes_ctc)
            char_list: Mapping from class index (1-based) to character string
        Returns:
            List of decoded strings for each sample in batch.
        """
        # Argmax along class dimension -> (T, B)
        _, preds = torch.max(log_probs, dim=2)
        preds = preds.permute(1, 0)  # Shape: (B, T)

        decoded_results = []
        for batch_idx in range(preds.size(0)):
            seq = preds[batch_idx].cpu().numpy()
            decoded_chars = []
            prev_idx = -1

            for idx in seq:
                # 0 is the CTC blank token
                if idx != 0 and idx != prev_idx:
                    # Index 1-47 correspond to character list index 0-46
                    char_idx = idx - 1
                    if 0 <= char_idx < len(char_list):
                        decoded_chars.append(char_list[char_idx])
                prev_idx = idx

            decoded_results.append("".join(decoded_chars))

        return decoded_results
