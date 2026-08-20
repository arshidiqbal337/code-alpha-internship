"""
Speech Emotion Recognition CLI Predictor
----------------------------------------
Classifies human emotion from a given audio file (.wav) using trained models.
Prints predicted emotion, confidence rating, and ASCII probability bar distribution.
"""

import os
import argparse
import numpy as np

from features import AudioFeatureExtractor
from models import EmotionClassifierEngine
from train import train_and_evaluate


def predict_audio_file(audio_path, model_path="emotion_recognition_model.pkl"):
    """
    Predicts human emotion from an audio file.
    Returns (top_emotion, confidence_percentage, emotion_probability_dict).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' does not exist.")

    # Check if model exists, train auto-fallback if needed
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Training default model first...")
        train_and_evaluate(data_path="sample_data", model_type="mlp", model_output=model_path)

    # Load model engine
    engine = EmotionClassifierEngine.load(model_path)
    extractor = AudioFeatureExtractor()

    # Load and extract features
    audio, sr = extractor.load_audio(audio_path)
    feat_vector = extractor.extract_feature_vector(audio, sr=sr)

    # Predict
    top_emotion, confidence, prob_dict = engine.predict_emotion(feat_vector)
    return top_emotion, confidence, prob_dict


def print_prediction_summary(audio_path, top_emotion, confidence, prob_dict):
    """Prints a clean ASCII prediction dashboard to the terminal."""
    label_map = {
        'happy': '[HAPPY]',
        'angry': '[ANGRY]',
        'sad': '[SAD]',
        'neutral': '[NEUTRAL]',
        'fearful': '[FEARFUL]',
        'disgust': '[DISGUST]',
        'surprised': '[SURPRISED]'
    }

    print("\n" + "=" * 60)
    print("       SPEECH EMOTION RECOGNITION (SER) - PREDICTION        ")
    print("=" * 60)
    print(f"  Target File : {os.path.basename(audio_path)}")
    print(f"  Result      : {label_map.get(top_emotion, top_emotion.upper())}")
    print(f"  Confidence  : {confidence:.2f}%")
    print("-" * 60)
    print("  EMOTION PROBABILITY BREAKDOWN:")
    print("-" * 60)

    # Sort by probability descending
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    for emo, prob in sorted_probs:
        bar_len = int(prob * 30)
        bar_str = "#" * bar_len + "-" * (30 - bar_len)
        lbl = emo.capitalize().ljust(10)
        pct = f"{prob * 100:5.1f}%"
        marker = "<-- (Top)" if emo == top_emotion else ""
        print(f"  {lbl} | {bar_str} | {pct} {marker}")

    print("=" * 60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Emotion from Speech Audio File")
    parser.add_argument("--file", type=str, required=True, help="Path to input .wav audio file")
    parser.add_argument("--model", type=str, default="emotion_recognition_model.pkl", help="Path to trained model .pkl")

    args = parser.parse_args()
    top_emo, conf, prob_map = predict_audio_file(args.file, model_path=args.model)
    print_prediction_summary(args.file, top_emo, conf, prob_map)
