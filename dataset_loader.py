"""
Dataset Loader and Speech Synthesizer Module
---------------------------------------------
Provides data loaders for standard Speech Emotion Recognition (SER) datasets:
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- TESS (Toronto Emotional Speech Set)
- EMO-DB (Berlin Database of Emotional Speech)

Includes a Synthetic Speech Generator to create realistic benchmark audio files
for immediate offline execution and zero-dependency testing out of the box.
"""

import os
import glob
import math
import numpy as np
from scipy.io import wavfile
from features import AudioFeatureExtractor

# Unified Emotion Categories (7 core classes)
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
EMOTION_TO_ID = {emo: i for i, emo in enumerate(EMOTIONS)}
ID_TO_EMOTION = {i: emo for i, emo in enumerate(EMOTIONS)}

# RAVDESS Emotion Code mapping
RAVDESS_MAP = {
    '01': 'neutral',
    '02': 'neutral',  # calm mapped to neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# EMO-DB Emotion Code mapping
EMODB_MAP = {
    'W': 'angry',
    'L': 'neutral',  # bored -> neutral
    'E': 'disgust',
    'F': 'fearful',
    'A': 'happy',
    'M': 'sad',
    'N': 'neutral'
}


class SyntheticSpeechGenerator:
    """Generates realistic synthetic audio files representing different human speech emotions."""

    def __init__(self, sample_rate=22050, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)

    def generate_emotion_audio(self, emotion):
        """
        Synthesizes a speech-like acoustic audio wave with emotion-specific prosody:
        - Pitch (F0 base frequency)
        - Harmonic richness & formants
        - Dynamic amplitude envelope & vibrato
        - Tremor and noise content
        """
        t = np.linspace(0, self.duration, self.num_samples)

        # Baseline vowel formant frequencies (F1, F2, F3)
        if emotion == 'angry':
            base_f0 = 260.0  # High pitch
            f0_std = 45.0    # Wild pitch swings
            vibrato_rate = 7.0
            volume_scale = 0.9
            harmonic_weights = [1.0, 0.8, 0.6, 0.5, 0.4]
            breathiness = 0.05
        elif emotion == 'happy':
            base_f0 = 230.0  # Upbeat high pitch
            f0_std = 35.0    # Dynamic pitch contours
            vibrato_rate = 5.5
            volume_scale = 0.8
            harmonic_weights = [1.0, 0.7, 0.5, 0.3, 0.2]
            breathiness = 0.03
        elif emotion == 'sad':
            base_f0 = 130.0  # Low pitch
            f0_std = 10.0    # Flat pitch contour
            vibrato_rate = 2.0
            volume_scale = 0.45
            harmonic_weights = [1.0, 0.3, 0.1, 0.05, 0.02]
            breathiness = 0.08
        elif emotion == 'fearful':
            base_f0 = 240.0  # High tense pitch
            f0_std = 30.0    # Jittery trembling
            vibrato_rate = 9.0
            volume_scale = 0.65
            harmonic_weights = [1.0, 0.6, 0.4, 0.3, 0.2]
            breathiness = 0.12
        elif emotion == 'disgust':
            base_f0 = 150.0  # Low creaky voice
            f0_std = 15.0
            vibrato_rate = 3.0
            volume_scale = 0.55
            harmonic_weights = [1.0, 0.5, 0.4, 0.3, 0.1]
            breathiness = 0.10
        elif emotion == 'surprised':
            base_f0 = 280.0  # Sharp upward pitch spike
            f0_std = 60.0
            vibrato_rate = 6.0
            volume_scale = 0.85
            harmonic_weights = [1.0, 0.75, 0.5, 0.3, 0.2]
            breathiness = 0.04
        else:  # neutral
            base_f0 = 175.0  # Moderate pitch
            f0_std = 15.0
            vibrato_rate = 4.0
            volume_scale = 0.6
            harmonic_weights = [1.0, 0.5, 0.3, 0.15, 0.05]
            breathiness = 0.04

        # Speech envelope: attack, sustain, decay pulses mimicking spoken words
        num_syllables = np.random.randint(3, 6)
        envelope = np.zeros_like(t)
        syl_len = len(t) // num_syllables
        for s in range(num_syllables):
            st = s * syl_len
            en = min(len(t), (s + 1) * syl_len)
            win = np.hanning(en - st)
            envelope[st:en] += win

        # Pitch modulation over time (speech intonation curve)
        f0_curve = base_f0 + f0_std * np.sin(2 * np.pi * (0.8 + 0.3 * np.random.rand()) * t)
        f0_curve += (vibrato_rate * np.sin(2 * np.pi * vibrato_rate * t))

        # Synthesize harmonic wave (fundamental + overtones)
        phase = 2 * np.pi * np.cumsum(f0_curve) / self.sample_rate
        audio_signal = np.zeros_like(t)
        for h, w in enumerate(harmonic_weights, start=1):
            audio_signal += w * np.sin(h * phase)

        # Add speech noise/breathiness component
        noise = np.random.normal(0, breathiness, size=len(t))
        audio_signal = (audio_signal + noise) * envelope * volume_scale

        # Normalize float range [-1.0, 1.0]
        audio_signal = audio_signal / (np.max(np.abs(audio_signal)) + 1e-6)
        return audio_signal.astype(np.float32)

    def create_sample_dataset(self, output_dir="sample_data", samples_per_emotion=15):
        """
        Creates a synthetic dataset directory with WAV files for each emotion class.
        Returns total files created.
        """
        os.makedirs(output_dir, exist_ok=True)
        total_created = 0

        for emotion in EMOTIONS:
            emo_dir = os.path.join(output_dir, emotion)
            os.makedirs(emo_dir, exist_ok=True)

            for idx in range(samples_per_emotion):
                file_path = os.path.join(emo_dir, f"{emotion}_sample_{idx+1:02d}.wav")
                audio = self.generate_emotion_audio(emotion)
                # Save as 16-bit PCM WAV file
                int_data = (audio * 32767.0).astype(np.int16)
                wavfile.write(file_path, self.sample_rate, int_data)
                total_created += 1

        print(f"Created {total_created} synthetic sample audio files in '{output_dir}/'.")
        return total_created


class DatasetLoader:
    """Loads and indexes speech dataset audio files (RAVDESS, TESS, EMO-DB, or Synthetic)."""

    def __init__(self, data_path="sample_data", extractor=None):
        self.data_path = data_path
        self.extractor = extractor or AudioFeatureExtractor()

    def parse_dataset_files(self):
        """
        Scans dataset directory for audio files and identifies emotion labels.
        Returns list of (file_path, emotion_label).
        """
        file_label_pairs = []

        if not os.path.exists(self.data_path):
            print(f"Directory '{self.data_path}' not found. Generating synthetic dataset...")
            gen = SyntheticSpeechGenerator()
            gen.create_sample_dataset(output_dir=self.data_path, samples_per_emotion=15)

        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if not file.lower().endswith(('.wav', '.mp3', '.flac')):
                    continue
                file_path = os.path.join(root, file)
                emotion = self._detect_emotion_from_filename(file_path)
                if emotion in EMOTION_TO_ID:
                    file_label_pairs.append((file_path, emotion))

        if len(file_label_pairs) == 0:
            print("No valid files found. Creating synthetic fallback samples...")
            gen = SyntheticSpeechGenerator()
            gen.create_sample_dataset(output_dir=self.data_path, samples_per_emotion=15)
            return self.parse_dataset_files()

        return file_label_pairs

    def _detect_emotion_from_filename(self, file_path):
        """Identifies emotion from folder structure or RAVDESS / TESS / EMO-DB naming standard."""
        file_name = os.path.basename(file_path).lower()
        folder_name = os.path.basename(os.path.dirname(file_path)).lower()

        # Check folder name first
        if folder_name in EMOTION_TO_ID:
            return folder_name

        # RAVDESS filename format: 03-01-05-01-02-01-02.wav
        parts = file_name.split('.')[0].split('-')
        if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
            return RAVDESS_MAP[parts[2]]

        # TESS filename format: YAF_angry.wav or OAF_happy.wav
        for emo in EMOTIONS:
            if emo in file_name:
                return emo
        if 'pleasant_surprised' in file_name or 'ps' in file_name:
            return 'surprised'

        # EMO-DB filename format: 03a01Wa.wav
        if len(file_name) >= 6 and file_name[5].upper() in EMODB_MAP:
            return EMODB_MAP[file_name[5].upper()]

        return 'neutral'

    def load_dataset_features(self, feature_type='aggregated'):
        """
        Loads all dataset audio files and extracts feature arrays.
        :param feature_type: 'aggregated' (1D vector) or 'sequence' (2D matrix)
        :return: X (features), y (numeric labels 0-6), file_paths
        """
        file_pairs = self.parse_dataset_files()
        X_list, y_list, valid_paths = [], [], []

        print(f"Extracting features from {len(file_pairs)} dataset files...")
        for path, emo in file_pairs:
            try:
                audio, sr = self.extractor.load_audio(path)
                if feature_type == 'aggregated':
                    feat = self.extractor.extract_feature_vector(audio, sr=sr)
                else:
                    feat = self.extractor.extract_sequence_features(audio, sr=sr)

                X_list.append(feat)
                y_list.append(EMOTION_TO_ID[emo])
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        X = np.array(X_list)
        y = np.array(y_list)
        return X, y, valid_paths


# Quick self-test script
if __name__ == '__main__':
    print("Testing SyntheticSpeechGenerator & DatasetLoader...")
    gen = SyntheticSpeechGenerator()
    gen.create_sample_dataset("sample_data", samples_per_emotion=5)

    loader = DatasetLoader("sample_data")
    X, y, paths = loader.load_dataset_features(feature_type='aggregated')

    print(f"Extracted dataset shape: X={X.shape}, y={y.shape}")
    print(f"Emotion distribution: {np.bincount(y)}")
    print("DatasetLoader self-test completed successfully!")
