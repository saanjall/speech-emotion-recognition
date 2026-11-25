"""
audio_predictor.py - load ANY trained audio model and return 8 emotion probabilities.

    pred = AudioEmotionPredictor("cnn_bilstm")
    probs = pred.predict_file("clip.wav")            # np.array, shape (8,), sums to 1
    probs = pred.predict_waveform(mic_audio, sr=44100)

Whatever the input sample rate, the audio is resampled to the model's rate and passed
through the SAME data.prepare_waveform() used during training.
"""
import os
import json

import numpy as np
import librosa
import joblib

import config
import data


class AudioEmotionPredictor:
    def __init__(self, model_name="cnn_bilstm", models_dir=config.MODELS_DIR):
        if model_name not in config.AUDIO_MODELS:
            raise ValueError(f"model_name must be one of {config.AUDIO_MODELS}")
        self.model_name = model_name
        self.dir = os.path.join(models_dir, model_name)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"No trained model in {self.dir}. Train it first (see README).")
        self.class_names = config.EMOTIONS

        if model_name in config.TORCH_AUDIO_MODELS:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.dir)
            self.model = AutoModelForAudioClassification.from_pretrained(self.dir).to(self.device).eval()
            self.sr = config.W2V_SR
        else:
            import models                                    # imports TensorFlow only when needed
            import keras
            self.to_numpy = keras.ops.convert_to_numpy         # works for TF tensors and CUDA torch tensors
            self.model = models.load_keras_model(os.path.join(self.dir, "model.keras"))
            with open(os.path.join(self.dir, "meta.json")) as f:
                self.meta = json.load(f)
            self.sr = self.meta.get("sr", config.SR)
            if model_name == "mlp":
                self.scaler = joblib.load(os.path.join(self.dir, "scaler.joblib"))
            else:
                self.norm = data.SpecNormalizer.load(os.path.join(self.dir, "norm.json"))

    # ------------------------------------------------------------------
    def predict_waveform(self, y, sr):
        """y: raw mono waveform at sample rate sr -> probabilities (8,)."""
        y = np.asarray(y, dtype=np.float32).flatten()
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        y = data.prepare_waveform(y, sr=self.sr)

        if self.model_name in config.TORCH_AUDIO_MODELS:
            inputs = self.feature_extractor(y, sampling_rate=self.sr, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(inputs["input_values"].to(self.device)).logits
            return self.torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)

        if self.model_name == "mlp":
            x = self.scaler.transform(data.extract_handcrafted(y, self.sr)[None, :])
        else:
            x = self.norm.transform(data.extract_logmel(y, self.sr)[None, ...])[..., None]
        probs = self.to_numpy(self.model(x.astype(np.float32), training=False))[0]   # direct call: fast for 1 sample
        return probs.astype(np.float32)

    def predict_file(self, path):
        y, sr = librosa.load(path, sr=None, mono=True)
        return self.predict_waveform(y, sr)

    def label(self, probs):
        return self.class_names[int(np.argmax(probs))]
