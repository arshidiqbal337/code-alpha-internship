"""
Speech Signal Processing and Feature Extraction Module
-------------------------------------------------------
Provides high-performance audio signal analysis tools:
- MFCC (Mel-Frequency Cepstral Coefficients) + Deltas
- Chroma STFT (12 pitch classes)
- Log Mel-Spectrogram
- Spectral Contrast & Centroid
- Zero Crossing Rate (ZCR)
- Root Mean Square (RMS) Energy

Supports pure NumPy/SciPy execution with optional Librosa integration.
"""

import math
import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
import scipy.signal as signal
import os

# Try importing librosa if available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioFeatureExtractor:
    """Class to load speech audio signals and extract emotion-discriminative features."""

    def __init__(self, sample_rate=22050, duration=3.0, n_mfcc=40, n_mels=128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def load_audio(self, audio_source, sr=None):
        """
        Loads audio from file path or raw numpy array.
        Normalizes amplitude to [-1, 1] and resamples/pads to target duration.
        """
        if sr is None:
            sr = self.sample_rate

        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                raise FileNotFoundError(f"Audio file not found: {audio_source}")

            if LIBROSA_AVAILABLE:
                try:
                    y, loaded_sr = librosa.load(audio_source, sr=sr, mono=True, duration=self.duration)
                except Exception:
                    loaded_sr, y = self._load_wav_scipy(audio_source)
            else:
                loaded_sr, y = self._load_wav_scipy(audio_source)
        elif isinstance(audio_source, np.ndarray):
            y = audio_source.astype(np.float32)
            loaded_sr = sr
        else:
            raise ValueError("audio_source must be a file path or numpy array.")

        # Ensure 1D mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Normalize float range [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # Resample if sample rate differs
        if loaded_sr != sr and len(y) > 0:
            num_samples = int(len(y) * sr / loaded_sr)
            y = signal.resample(y, num_samples)

        # Pad or trim to target length
        if len(y) < self.target_length:
            pad_width = self.target_length - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        elif len(y) > self.target_length:
            y = y[:self.target_length]

        return y, sr

    def _load_wav_scipy(self, file_path):
        """Fallback loader using scipy.io.wavfile"""
        try:
            sr, data = wavfile.read(file_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            else:
                data = data.astype(np.float32)
            return sr, data
        except Exception as e:
            # Handle synthetically generated or headerless files
            raise IOError(f"Could not read WAV file {file_path}: {e}")

    # -------------------------------------------------------------
    # Signal Processing Utility Methods (NumPy/SciPy Implementations)
    # -------------------------------------------------------------

    def hz_to_mel(self, hz):
        """Convert Hz frequency to Mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(self, mel):
        """Convert Mel scale to Hz frequency."""
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    def get_mel_filterbank(self, sr, n_fft=1024, n_mels=128):
        """Construct Mel filter bank matrix."""
        num_freqs = int(n_fft // 2 + 1)
        low_mel = self.hz_to_mel(0)
        high_mel = self.hz_to_mel(sr / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((n_mels, num_freqs))
        for m in range(1, n_mels + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return fbank

    def stft(self, y, n_fft=1024, hop_length=512):
        """Short-Time Fourier Transform (STFT)."""
        window = np.hanning(n_fft)
        num_frames = max(1, int((len(y) - n_fft) // hop_length + 1))
        stft_matrix = np.zeros((num_frames, int(n_fft // 2 + 1)), dtype=np.complex64)

        for i in range(num_frames):
            start = i * hop_length
            end = start + n_fft
            if end > len(y):
                frame = np.pad(y[start:], (0, end - len(y)), mode='constant')
            else:
                frame = y[start:end]
            frame_win = frame * window
            fft_frame = np.fft.rfft(frame_win, n=n_fft)
            stft_matrix[i] = fft_frame

        return stft_matrix

    # -------------------------------------------------------------
    # Feature Extractors
    # -------------------------------------------------------------

    def extract_mfcc(self, y, sr=22050, n_mfcc=40, n_fft=1024, hop_length=512):
        """Extract Mel-Frequency Cepstral Coefficients (MFCCs)."""
        if LIBROSA_AVAILABLE:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            return mfcc

        # Pure NumPy/SciPy implementation
        stft_matrix = self.stft(y, n_fft=n_fft, hop_length=hop_length)
        pow_frames = np.abs(stft_matrix) ** 2
        fbank = self.get_mel_filterbank(sr, n_fft=n_fft, n_mels=self.n_mels)
        mel_spectrogram = np.dot(pow_frames, fbank.T)
        mel_spectrogram = np.where(mel_spectrogram == 0, np.finfo(float).eps, mel_spectrogram)
        log_mel = np.log(mel_spectrogram)
        mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
        return mfcc.T  # Shape: (n_mfcc, time_steps)

    def extract_chroma(self, y, sr=22050, n_fft=1024, hop_length=512):
        """Extract Chroma STFT feature map (12 pitch classes)."""
        if LIBROSA_AVAILABLE:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            return chroma

        stft_matrix = np.abs(self.stft(y, n_fft=n_fft, hop_length=hop_length)).T
        num_freqs, num_frames = stft_matrix.shape
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        chroma = np.zeros((12, num_frames))
        # Map frequencies to 12 semitones relative to A4 (440 Hz)
        valid_idx = freqs > 0
        valid_freqs = freqs[valid_idx]
        pitch_bins = (12 * np.log2(valid_freqs / 440.0) + 69) % 12
        pitch_bins = np.round(pitch_bins).astype(int) % 12

        for i, bin_idx in enumerate(pitch_bins):
            chroma[bin_idx, :] += stft_matrix[valid_idx][i, :]

        # Normalize chroma
        norm = np.linalg.norm(chroma, axis=0, keepdims=True)
        norm[norm == 0] = 1.0
        return chroma / norm

    def extract_mel_spectrogram(self, y, sr=22050, n_mels=128, n_fft=1024, hop_length=512):
        """Extract Log Mel Spectrogram matrix."""
        if LIBROSA_AVAILABLE:
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            return librosa.power_to_db(mel_spec, ref=np.max)

        stft_matrix = self.stft(y, n_fft=n_fft, hop_length=hop_length)
        pow_frames = np.abs(stft_matrix) ** 2
        fbank = self.get_mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels)
        mel_spec = np.dot(pow_frames, fbank.T).T
        mel_spec = np.where(mel_spec <= 0, 1e-10, mel_spec)
        log_mel_spec = 10.0 * np.log10(mel_spec / np.max(mel_spec))
        return log_mel_spec

    def extract_zcr(self, y, frame_length=1024, hop_length=512):
        """Extract Zero Crossing Rate (ZCR)."""
        num_frames = max(1, int((len(y) - frame_length) // hop_length + 1))
        zcr = np.zeros((1, num_frames))
        for i in range(num_frames):
            start = i * hop_length
            frame = y[start:start + frame_length]
            zcr[0, i] = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        return zcr

    def extract_rms(self, y, frame_length=1024, hop_length=512):
        """Extract Root Mean Square (RMS) Energy."""
        num_frames = max(1, int((len(y) - frame_length) // hop_length + 1))
        rms = np.zeros((1, num_frames))
        for i in range(num_frames):
            start = i * hop_length
            frame = y[start:start + frame_length]
            rms[0, i] = np.sqrt(np.mean(frame**2))
        return rms

    def compute_deltas(self, feature_matrix, N=2):
        """Compute delta features along time axis (2nd dimension)."""
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(-1, 1)

        rows, cols = feature_matrix.shape
        delta = np.zeros_like(feature_matrix)

        for col in range(cols):
            denom = 2 * sum([i**2 for i in range(1, N + 1)])
            for n in range(1, N + 1):
                prev_c = max(0, col - n)
                next_c = min(cols - 1, col + n)
                delta[:, col] += n * (feature_matrix[:, next_c] - feature_matrix[:, prev_c])
            delta[:, col] /= denom

        return delta

    def extract_feature_vector(self, y, sr=22050):
        """
        Extracts aggregated 1D feature vector summarizing spectral and prosodic statistics.
        Returns a 1D numpy array ideal for traditional ML & MLP models.
        """
        mfcc = self.extract_mfcc(y, sr=sr, n_mfcc=self.n_mfcc)
        delta_mfcc = self.compute_deltas(mfcc)
        chroma = self.extract_chroma(y, sr=sr)
        mel_spec = self.extract_mel_spectrogram(y, sr=sr, n_mels=64)
        zcr = self.extract_zcr(y)
        rms = self.extract_rms(y)

        # Aggregate statistical features (mean, std, max, min) across time frames
        def get_stats(mat):
            return np.hstack([
                np.mean(mat, axis=1),
                np.std(mat, axis=1),
                np.max(mat, axis=1),
                np.min(mat, axis=1)
            ])

        feat_vector = np.hstack([
            get_stats(mfcc),
            get_stats(delta_mfcc),
            get_stats(chroma),
            get_stats(mel_spec),
            get_stats(zcr),
            get_stats(rms)
        ])

        return feat_vector

    def extract_sequence_features(self, y, sr=22050):
        """
        Extracts 2D feature matrix (n_features, time_steps) suitable for 1D-CNN and LSTM models.
        """
        mfcc = self.extract_mfcc(y, sr=sr, n_mfcc=self.n_mfcc)
        chroma = self.extract_chroma(y, sr=sr)
        zcr = self.extract_zcr(y)
        rms = self.extract_rms(y)

        # Concatenate along feature dimension
        seq_features = np.vstack([mfcc, chroma, zcr, rms])
        return seq_features  # Shape: (54, time_steps)


# Quick self-test script
if __name__ == '__main__':
    print("Testing AudioFeatureExtractor...")
    extractor = AudioFeatureExtractor()
    # Generate 3 seconds of dummy audio (440Hz sine wave with noise)
    t = np.linspace(0, 3.0, int(22050 * 3.0))
    dummy_audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

    y, sr = extractor.load_audio(dummy_audio, sr=22050)
    feat_vec = extractor.extract_feature_vector(y, sr=sr)
    seq_feat = extractor.extract_sequence_features(y, sr=sr)
    mel_spec = extractor.extract_mel_spectrogram(y, sr=sr)

    print(f"Loaded audio length: {len(y)} samples at {sr} Hz")
    print(f"Aggregated Feature Vector shape: {feat_vec.shape}")
    print(f"Sequence Feature Matrix shape: {seq_feat.shape}")
    print(f"Mel Spectrogram shape: {mel_spec.shape}")
    print("Feature Extraction self-test passed successfully!")
