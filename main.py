"""
Master Entry Point - Emotion Recognition from Speech (SER)
-----------------------------------------------------------
Pure Python Command Center for Speech Emotion Recognition.

Usage:
  python main.py                              # Launch Native Desktop GUI Studio
  python main.py --gui                        # Launch Native Desktop GUI Studio
  python main.py --train                      # Train Emotion Model on Dataset
  python main.py --predict <audio_path>       # Predict Emotion from Audio WAV file
  python main.py --generate-samples           # Synthesize Benchmark Sample WAV files
"""

import sys
import argparse
import os

from dataset_loader import SyntheticSpeechGenerator, DatasetLoader
from train import train_and_evaluate
from predict import predict_audio_file, print_prediction_summary


def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition System (Pure Python)")
    parser.add_argument("--gui", action="store_true", help="Launch Native Desktop GUI Studio")
    parser.add_argument("--train", action="store_true", help="Train Emotion Classifier Model")
    parser.add_argument("--predict", type=str, metavar="AUDIO_FILE", help="Predict emotion for an audio .wav file")
    parser.add_argument("--generate-samples", action="store_true", help="Generate synthetic benchmark audio dataset")
    parser.add_argument("--data-path", type=str, default="sample_data", help="Dataset folder path")
    parser.add_argument("--model-type", type=str, default="mlp", choices=["mlp", "ensemble"], help="Model type")
    parser.add_argument("--model-path", type=str, default="emotion_recognition_model.pkl", help="Model file path")

    args = parser.parse_args()

    # 1. Synthesize samples if requested
    if args.generate_samples:
        print(f"Generating synthetic speech emotion samples in '{args.data_path}'...")
        gen = SyntheticSpeechGenerator()
        gen.create_sample_dataset(args.data_path, samples_per_emotion=15)
        return

    # 2. Train model if requested
    if args.train:
        print("Starting Speech Emotion Recognition model training...")
        train_and_evaluate(data_path=args.data_path, model_type=args.model_type, model_output=args.model_path)
        return

    # 3. Predict emotion if file specified
    if args.predict:
        top_emo, conf, prob_map = predict_audio_file(args.predict, model_path=args.model_path)
        print_prediction_summary(args.predict, top_emo, conf, prob_map)
        return

    # 4. Default: Launch GUI App
    print("Launching Speech Emotion Recognition Studio (Native Desktop UI)...")
    try:
        from gui_app import SpeechEmotionGUI
        app = SpeechEmotionGUI(model_path=args.model_path)
        app.mainloop()
    except Exception as e:
        print(f"Note: Desktop GUI window closed or could not display display server: {e}")
        print("You can run CLI commands using: python main.py --predict sample_data/angry/angry_sample_01.wav")


if __name__ == '__main__':
    main()
