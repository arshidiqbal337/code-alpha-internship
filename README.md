# Handwritten Character and Word Recognition System in Python

A clean, production-grade Python system for **Handwritten Character, Digit, and Word Recognition** built using **PyTorch**, **OpenCV**, and **NumPy**.

## Overview & Key Features

- **Datasets Supported**:
  - **MNIST**: 10 classes (digits `0-9`).
  - **EMNIST (Balanced)**: 47 classes (`0-9`, `A-Z`, and `a, b, d, e, f, g, h, n, q, r, t`). Corrects the EMNIST transpose/rotation issue.
- **Deep Learning Architectures**:
  - **CharacterCNN (`models/cnn_model.py`)**: 4-layer Deep Convolutional Neural Network with Batch Normalization, Max Pooling, and Dropout for high-accuracy isolated character classification.
  - **WordCRNN (`models/crnn_model.py`)**: Convolutional Recurrent Neural Network combining CNN feature extraction, Map-to-Sequence layer, 2-layer Bidirectional LSTM, and Connectionist Temporal Classification (CTC) loss decoder for continuous text recognition.
- **Image Processing & Segmentation (`utils/preprocessing.py`)**:
  - Otsu thresholding & adaptive binarization with automatic background inversion.
  - Aspect-ratio preserving resize and symmetric padding to $28 \times 28$.
  - OpenCV contour-based bounding box extraction for multi-character word/sentence segmentation.
- **Clean CLI & API**:
  - No GUI/UI required. Command line interfaces for training (`train.py`) and inference (`predict.py`).
  - Modular, well-documented code with type hints and docstrings.

---

## Project Structure

```
project 3/
├── config.py                  # Global hyperparameters, data paths, and EMNIST mapping
├── requirements.txt           # Python dependencies
├── train.py                   # Script to train CNN and CRNN models
├── predict.py                 # CLI inference for single character and word images
├── sample_demo.py             # Self-contained demo script (runnable out-of-the-box)
├── models/
│   ├── __init__.py
│   ├── cnn_model.py           # Deep CNN architecture for character classification
│   └── crnn_model.py          # CRNN architecture (CNN + BiLSTM + CTC Loss)
├── utils/
│   ├── __init__.py
│   ├── dataset.py             # MNIST/EMNIST dataset loaders & synthetic word generator
│   ├── preprocessing.py       # Thresholding, padding, and character contour segmentation
│   └── visualization.py       # Plotting training curves & drawing annotated bounding boxes
└── README.md                  # Detailed documentation
```

---

## Installation

1. **Clone or navigate to the workspace**:
   ```bash
   cd "c:\Users\AL AZIZ\OneDrive\Desktop\Decode Internship\project 3"
   ```

2. **Install Required Python Packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Run End-to-End Demo Script
To run a complete test of dataset loading, model training, character prediction, and word segmentation without any setup:
```bash
python sample_demo.py
```
This will generate synthetic test images, train a sample model, segment handwritten words into characters, and save visual output artifacts in the `outputs/` directory.

---

### 2. Training Models

- **Train CNN on EMNIST (47 classes)**:
  ```bash
  python train.py --model cnn --dataset emnist --epochs 10 --batch_size 64 --lr 0.001
  ```

- **Train CNN on MNIST (10 digit classes)**:
  ```bash
  python train.py --model cnn --dataset mnist --epochs 10
  ```

- **Train CRNN Sequence Model (Word Recognition with CTC Loss)**:
  ```bash
  python train.py --model crnn --epochs 15
  ```

Saved model checkpoints will be stored under `checkpoints/`.

---

### 3. Predicting / Inference

- **Predict a Single Isolated Character**:
  ```bash
  python predict.py --image "path/to/character.png" --mode single
  ```

- **Recognize a Full Handwritten Word (Contour Segmentation + CNN)**:
  ```bash
  python predict.py --image "path/to/word.png" --mode word --output "outputs/word_result.png"
  ```

- **Recognize a Continuous Word Image using CRNN Model**:
  ```bash
  python predict.py --image "path/to/word.png" --mode crnn
  ```

---

## Technical Details & Architecture

### 1. Character CNN Model Architecture
- **Conv Block 1**: `Conv2d(1, 32)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(32, 64)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2x2)` $\rightarrow$ `Dropout(0.25)`
- **Conv Block 2**: `Conv2d(64, 128)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2d(128, 128)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2x2)` $\rightarrow$ `Dropout(0.25)`
- **Fully-Connected Head**: `Linear(128 * 7 * 7, 256)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(256, 47)`

### 2. CRNN Sequence Architecture
- **CNN Feature Extractor**: Reduces height from 32 down to 1 while extracting spatial features across sequence width.
- **BiLSTM Sequence Model**: 2-layer Bidirectional LSTM processing sequence frames along the horizontal axis.
- **CTC Loss Decoding**: CTC (Connectionist Temporal Classification) blank token collapsing for alignment-free word decoding.
