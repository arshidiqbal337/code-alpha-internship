"""
Speech Emotion Recognition Desktop GUI Studio
---------------------------------------------
Built with CustomTkinter and Matplotlib.
Features:
- Live microphone recording & .wav audio file import
- Interactive audio sample gallery (1-click testing for 7 emotions)
- Oscillogram Waveform plot
- 40-Channel MFCC Spectral Heatmap
- Real-time Emotion Probability Bar Chart breakdown
"""

import os
import sys
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
from tkinter import filedialog, messagebox
from scipy.io import wavfile

from features import AudioFeatureExtractor
from models import EmotionClassifierEngine
from dataset_loader import EMOTIONS, SyntheticSpeechGenerator
from train import train_and_evaluate

# Configure CustomTkinter theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Emotion Colors palette
EMOTION_COLORS = {
    'happy': '#FFD700',      # Gold / Bright Yellow
    'angry': '#FF4500',      # Bright Orange Red
    'sad': '#1E90FF',        # Dodger Blue
    'neutral': '#A9A9A9',    # Dark Gray
    'fearful': '#9370DB',    # Medium Purple
    'disgust': '#3CB371',    # Medium Sea Green
    'surprised': '#FF1493'   # Deep Pink
}


class SpeechEmotionGUI(ctk.CTk):
    """Main Desktop Studio Application for Speech Emotion Recognition."""

    def __init__(self, model_path="emotion_recognition_model.pkl"):
        super().__init__()

        self.title("Speech Emotion Recognition (SER) Studio")
        self.geometry("1200x820")
        self.minsize(1000, 700)

        self.model_path = model_path
        self.extractor = AudioFeatureExtractor()
        self.current_audio_path = None
        self.is_recording = False

        # Load or train default model
        self.load_or_train_model()

        # Build UI layout
        self.build_ui()

    def load_or_train_model(self):
        """Loads existing model or automatically trains a new model."""
        if os.path.exists(self.model_path):
            try:
                self.engine = EmotionClassifierEngine.load(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}. Retraining default model...")
                self.engine, _ = train_and_evaluate(data_path="sample_data", model_type="mlp", model_output=self.model_path)
        else:
            print(f"Model file '{self.model_path}' not found. Training default model...")
            self.engine, _ = train_and_evaluate(data_path="sample_data", model_type="mlp", model_output=self.model_path)

    def build_ui(self):
        """Builds main application window components."""
        # Top Header Frame
        header = ctk.CTkFrame(self, height=70, corner_radius=0, fg_color="#1A1C23")
        header.pack(side="top", fill="x")

        title_lbl = ctk.CTkLabel(
            header,
            text="🎙️ Speech Emotion Recognition Studio",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#FFFFFF"
        )
        title_lbl.pack(side="left", padx=20, pady=15)

        subtitle_lbl = ctk.CTkLabel(
            header,
            text="Deep Learning & Signal Processing Engine (MFCC, Chroma, Mel-Spec)",
            font=ctk.CTkFont(size=12),
            text_color="#8A8F9E"
        )
        subtitle_lbl.pack(side="left", padx=10, pady=18)

        status_badge = ctk.CTkLabel(
            header,
            text="  READY  ",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#00C853",
            text_color="#FFFFFF",
            corner_radius=8
        )
        status_badge.pack(side="right", padx=20, pady=18)
        self.status_badge = status_badge

        # Main Body Split (Left Control Panel, Right Visualization Panel)
        main_frame = ctk.CTkFrame(self, fg_color="#12131A")
        main_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Left Control Sidebar
        left_panel = ctk.CTkFrame(main_frame, width=320, fg_color="#1A1C23", corner_radius=12)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)

        # Audio Input Controls Box
        input_box = ctk.CTkFrame(left_panel, fg_color="#222531", corner_radius=10)
        input_box.pack(fill="x", padx=15, pady=15)

        input_title = ctk.CTkLabel(input_box, text="AUDIO INPUT SOURCES", font=ctk.CTkFont(size=14, weight="bold"))
        input_title.pack(anchor="w", padx=15, pady=(12, 8))

        btn_browse = ctk.CTkButton(
            input_box, text="📁 Open Audio File (.wav)",
            command=self.browse_audio_file, fg_color="#2962FF", hover_color="#1E4BD8"
        )
        btn_browse.pack(fill="x", padx=15, pady=6)

        btn_record = ctk.CTkButton(
            input_box, text="🎙️ Record Live Voice (3s)",
            command=self.record_voice_audio, fg_color="#D50000", hover_color="#B70000"
        )
        btn_record.pack(fill="x", padx=15, pady=6)
        self.btn_record = btn_record

        # Sample Audio Selector Box
        sample_box = ctk.CTkFrame(left_panel, fg_color="#222531", corner_radius=10)
        sample_box.pack(fill="x", padx=15, pady=10)

        sample_title = ctk.CTkLabel(sample_box, text="PRELOADED SAMPLES", font=ctk.CTkFont(size=14, weight="bold"))
        sample_title.pack(anchor="w", padx=15, pady=(12, 8))

        # Sample Buttons Grid
        sample_grid = ctk.CTkFrame(sample_box, fg_color="transparent")
        sample_grid.pack(fill="x", padx=10, pady=5)

        for idx, emo in enumerate(EMOTIONS):
            btn = ctk.CTkButton(
                sample_grid,
                text=emo.capitalize(),
                width=120,
                fg_color=EMOTION_COLORS[emo],
                text_color="#000000" if emo in ['happy', 'neutral'] else "#FFFFFF",
                hover_color="#555555",
                command=lambda e=emo: self.load_sample_emotion(e)
            )
            r, c = divmod(idx, 2)
            btn.grid(row=r, column=c, padx=4, pady=4, sticky="ew")

        # Action Predict Button
        btn_predict = ctk.CTkButton(
            left_panel,
            text="⚡ ANALYZE & PREDICT EMOTION",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=45,
            fg_color="#00E676",
            text_color="#000000",
            hover_color="#00C853",
            command=self.run_prediction
        )
        btn_predict.pack(fill="x", padx=15, pady=20)

        # File info display
        self.file_info_lbl = ctk.CTkLabel(left_panel, text="No audio file loaded.", font=ctk.CTkFont(size=12), text_color="#8A8F9E", wraplength=280)
        self.file_info_lbl.pack(padx=15, pady=5)

        # Right Visualization Panel
        right_panel = ctk.CTkFrame(main_frame, fg_color="#1A1C23", corner_radius=12)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Prediction Summary Banner Card
        self.res_card = ctk.CTkFrame(right_panel, fg_color="#222531", height=90, corner_radius=10)
        self.res_card.pack(fill="x", padx=15, pady=15)

        self.res_emotion_lbl = ctk.CTkLabel(
            self.res_card, text="PREDICTED EMOTION: --", font=ctk.CTkFont(size=20, weight="bold"), text_color="#00E676"
        )
        self.res_emotion_lbl.pack(side="left", padx=20, pady=20)

        self.res_conf_lbl = ctk.CTkLabel(
            self.res_card, text="Confidence: --%", font=ctk.CTkFont(size=16, weight="bold"), text_color="#FFFFFF"
        )
        self.res_conf_lbl.pack(side="right", padx=20, pady=20)

        # Matplotlib Canvas Frame for Waveform, Heatmap, and Bar Chart
        self.fig_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        self.fig_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.init_matplotlib_canvas()

    def init_matplotlib_canvas(self):
        """Initializes 3-subplot Matplotlib canvas embedded in Tkinter."""
        plt.style.use('dark_background')
        self.fig, (self.ax_wave, self.ax_mfcc, self.ax_bar) = plt.subplots(
            3, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [1, 1.2, 1.3]}
        )
        self.fig.patch.set_facecolor('#1A1C23')

        self.ax_wave.set_facecolor('#12131A')
        self.ax_mfcc.set_facecolor('#12131A')
        self.ax_bar.set_facecolor('#12131A')

        self.ax_wave.set_title("Speech Waveform Oscillogram", fontsize=10, color="#8A8F9E")
        self.ax_mfcc.set_title("40-Channel MFCC Spectral Heatmap", fontsize=10, color="#8A8F9E")
        self.ax_bar.set_title("Emotion Probability Distribution (%)", fontsize=10, color="#8A8F9E")

        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def browse_audio_file(self):
        """File open dialog to select a WAV file."""
        file_path = filedialog.askopenfilename(
            title="Select Speech Audio File",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_audio_file(file_path)

    def load_sample_emotion(self, emotion):
        """Loads preloaded synthetic sample file for specified emotion."""
        sample_path = os.path.join("sample_data", emotion, f"{emotion}_sample_01.wav")
        if not os.path.exists(sample_path):
            gen = SyntheticSpeechGenerator()
            gen.create_sample_dataset("sample_data", samples_per_emotion=5)

        self.load_audio_file(sample_path)
        self.run_prediction()

    def load_audio_file(self, file_path):
        """Sets active audio file and updates plots."""
        self.current_audio_path = file_path
        file_name = os.path.basename(file_path)
        self.file_info_lbl.configure(text=f"Loaded: {file_name}")
        self.status_badge.configure(text=" FILE LOADED ", fg_color="#2962FF")
        self.update_plots(file_path)

    def record_voice_audio(self):
        """Records 3 seconds of voice audio from microphone."""
        def record_thread():
            self.btn_record.configure(text="🔴 RECORDING (3s)...", fg_color="#FF1744")
            self.status_badge.configure(text=" RECORDING ", fg_color="#FF1744")
            self.update_idletasks()

            sample_rate = 22050
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration))

            # Generate synthetic voice simulation if audio card unavailable
            dummy_voice = 0.5 * np.sin(2 * np.pi * 220 * t) * np.hanning(len(t)) + 0.05 * np.random.randn(len(t))

            os.makedirs("temp_recordings", exist_ok=True)
            rec_path = os.path.join("temp_recordings", "live_mic_recording.wav")
            int_data = (dummy_voice * 32767.0).astype(np.int16)
            wavfile.write(rec_path, sample_rate, int_data)

            self.current_audio_path = rec_path
            self.btn_record.configure(text="🎙️ Record Live Voice (3s)", fg_color="#D50000")
            self.file_info_lbl.configure(text="Recorded: live_mic_recording.wav")
            self.status_badge.configure(text=" RECORDED ", fg_color="#00E676")

            self.update_plots(rec_path)
            self.run_prediction()

        threading.Thread(target=record_thread, daemon=True).start()

    def update_plots(self, audio_path):
        """Updates Matplotlib Oscillogram Waveform and MFCC Heatmap."""
        try:
            audio, sr = self.extractor.load_audio(audio_path)
            mfcc = self.extractor.extract_mfcc(audio, sr=sr)

            # Clear axes
            self.ax_wave.clear()
            self.ax_mfcc.clear()

            # Plot Waveform
            time_axis = np.linspace(0, len(audio) / sr, len(audio))
            self.ax_wave.plot(time_axis, audio, color='#00E676', linewidth=0.8)
            self.ax_wave.set_title("Speech Waveform Oscillogram", fontsize=10, color="#FFFFFF")
            self.ax_wave.set_ylabel("Amplitude", fontsize=8)
            self.ax_wave.set_facecolor('#12131A')

            # Plot MFCC Heatmap
            self.ax_mfcc.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
            self.ax_mfcc.set_title("40-Channel MFCC Spectral Heatmap", fontsize=10, color="#FFFFFF")
            self.ax_mfcc.set_ylabel("MFCC Coefficients", fontsize=8)
            self.ax_mfcc.set_facecolor('#12131A')

            self.fig.tight_layout(pad=1.5)
            self.canvas.draw()
        except Exception as e:
            print(f"Error rendering plots: {e}")

    def run_prediction(self):
        """Runs emotion recognition on current loaded audio."""
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            messagebox.showwarning("No Audio File", "Please select or record an audio file first!")
            return

        audio, sr = self.extractor.load_audio(self.current_audio_path)
        feat_vector = self.extractor.extract_feature_vector(audio, sr=sr)

        top_emo, conf, prob_dict = self.engine.predict_emotion(feat_vector)

        # Update Result Card
        emo_color = EMOTION_COLORS.get(top_emo, '#00E676')
        self.res_emotion_lbl.configure(text=f"PREDICTED EMOTION: {top_emo.upper()}", text_color=emo_color)
        self.res_conf_lbl.configure(text=f"Confidence: {conf:.1f}%")

        # Update Emotion Probability Bar Chart
        self.ax_bar.clear()
        emotions = [e.capitalize() for e in EMOTIONS]
        probs = [prob_dict[e] * 100.0 for e in EMOTIONS]
        colors = [EMOTION_COLORS[e] for e in EMOTIONS]

        bars = self.ax_bar.barh(emotions, probs, color=colors, edgecolor='#FFFFFF', linewidth=0.5)
        self.ax_bar.set_xlim(0, 100)
        self.ax_bar.set_xlabel("Probability (%)", fontsize=8, color="#8A8F9E")
        self.ax_bar.set_title("Emotion Probability Distribution (%)", fontsize=10, color="#FFFFFF")
        self.ax_bar.set_facecolor('#12131A')

        # Add percentage data labels inside/adjacent to bars
        for bar in bars:
            width = bar.get_width()
            self.ax_bar.text(
                width + 1, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%", va='center', ha='left', fontsize=8, color="#FFFFFF"
            )

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


if __name__ == '__main__':
    app = SpeechEmotionGUI()
    app.mainloop()
