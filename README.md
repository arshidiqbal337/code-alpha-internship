# Speech Emotion Recognition (SER) System

A pure Python deep learning and signal processing system for recognizing human emotions (e.g., **Happy**, **Angry**, **Sad**, **Neutral**, **Fearful**, **Disgust**, **Surprised**) from speech audio.

---

## Key Features

- **Signal Processing & Feature Extraction (`features.py`)**:
  - **40 MFCCs (Mel-Frequency Cepstral Coefficients)** & Delta MFCCs
  - **Chroma STFT** (12 pitch class distribution)
  - **Log Mel Spectrogram** (energy distribution across mel frequency scale)
  - **Zero Crossing Rate (ZCR)** & **Root Mean Square (RMS) Energy**
  - Audio amplitude normalization, silence trimming, and padding/framing routines

- **Model Architectures (`models.py`)**:
  - **1D-CNN**: Convolutional Neural Network operating over MFCC temporal frames
  - **2D-CNN**: Mel-Spectrogram image classifier
  - **Bi-LSTM**: Recurrent sequence architecture for speech temporal dynamics
  - **Multi-Layer Perceptron (MLP)**: Deep Neural Network core
  - **Ensemble Classifier**: Random Forest & Extra Trees ensemble

- **Supported Datasets & Synthesizer (`dataset_loader.py`)**:
  - **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
  - **TESS** (Toronto Emotional Speech Set)
  - **EMO-DB** (Berlin Database of Emotional Speech)
  - **Synthetic Speech Synthesizer**: Automatically generates 105 benchmark `.wav` audio files matching unique emotion prosody profiles for instant zero-dependency execution out of the box!

- **Native Python Desktop GUI Studio (`gui_app.py`)**:
  - Dark-mode user interface using `CustomTkinter`
  - Audio file browser (.wav) & 1-click sample emotion selector
  - Live microphone voice recording
  - Embedded **Matplotlib** visualizations:
    1. Speech Waveform Oscillogram
    2. 40-Channel MFCC Spectral Heatmap
    3. Emotion Probability Breakdown Horizontal Bar Chart

---

## File Overview

| File | Description |
|---|---|
| [`main.py`](file:///c:/Users/AL_AZIZ/OneDrive/Desktop/Decode_Internship/project_2/main.py) | Master entry point CLI controller (`--gui`, `--train`, `--predict`, `--generate-samples`) |
| [`gui_app.py`](file:///c:/Users/AL_AZIZ/OneDrive/Desktop/Decode_Internship/project_2/gui_app.py) | Native Python Desktop UI with live plots & audio player |
| [`predict.py`](file:///c:/Users/AL_AZIZ/OneDrive/Desktop/Decode_Internship/project_2/predict.py) | CLI prediction tool for `.wav` files with ASCII probability bar charts |
| [`train.py`](file:///c:/Users/AL_AZIZ/OneDrive/Desktop/Decode_Internship/project_2/train.py) | Model training script with classification metrics & confusion matrix plot |
| [`features.py`](file:///c:/Users/AL_AZIZ/OneDrive/Desktop/Decode_Internship/project_2/features.py) | Audio signal processing and feature extraction module (MFCCs, Chroma, Mel-Spec) |
| [`models.py`](file:///c:/Users/AL_AZIZ/OneDrive/Desktop/Decode_Internship/project_2/models.py) | Deep learning (PyTorch) and machine learning (Scikit-Learn) model architectures |
| [`dataset_loader.py`](file:///c:/Users/AL_AZIZ/OneDrive/Desktop/Decode_Internship/project_2/dataset_loader.py) | Dataset indexers for RAVDESS, TESS, EMO-DB, and synthetic speech generator |

---

## Quick Start & Usage

### 1. Launch Desktop GUI Studio
```bash
python main.py --gui
```
*or simply:*
```bash
python main.py
```

### 2. Predict Emotion for an Audio File
```bash
python main.py --predict sample_data/angry/angry_sample_01.wav
```

### 3. Train Model on Dataset
```bash
python main.py --train --model-type mlp
```

### 4. Generate Synthetic Benchmark Dataset
```bash
python main.py --generate-samples
```
