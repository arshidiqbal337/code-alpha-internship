"""
Global Configuration module for Handwritten Character and Word Recognition.
Contains dataset mappings, hyperparameters, and file paths.
"""

import os
import torch

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure directories exist
for directory in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Hardware Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Specifications
IMAGE_SIZE = (28, 28)       # Standard 28x28 for isolated character classification
CRNN_IMAGE_SIZE = (32, 128) # (Height, Width) for continuous word images in CRNN

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_CNN = 10
EPOCHS_CRNN = 15

# EMNIST Balanced Dataset Label Mapping (47 Classes)
# Index 0-9   : '0'-'9'
# Index 10-35 : 'A'-'Z'
# Index 36-46 : 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't' (EMNIST balanced subset of lowercase)
EMNIST_BALANCED_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

# Reverse mapping for string label to class index
CHAR_TO_INDEX = {v: k for k, v in EMNIST_BALANCED_MAPPING.items()}
NUM_CLASSES_EMNIST = len(EMNIST_BALANCED_MAPPING)

# MNIST Digits Dataset Mapping (10 Classes)
MNIST_MAPPING = {i: str(i) for i in range(10)}

# Characters list for CRNN sequence decoding (0-indexed CTC blank is handled separately)
CRNN_CHAR_LIST = [EMNIST_BALANCED_MAPPING[i] for i in range(NUM_CLASSES_EMNIST)]
