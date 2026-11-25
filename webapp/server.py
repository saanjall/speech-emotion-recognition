"""
webapp/server.py - "Feelspace" voice check-in app for student wellbeing.

Serves the front-end (webapp/static/) and a small JSON API around the trained
speech-emotion models in saved_models/.

    python webapp/server.py                 # http://localhost:5000
    python webapp/server.py --port 8000

API
    GET  /api/models    -> models found in saved_models/ (best first) + which one is default
    POST /api/analyze   -> multipart form: audio=<wav file>, model=<name (optional)>
                           returns emotion, confidence, all probabilities, a per-window
                           timeline and simple voice metrics (pitch, energy, pauses, pace)

Audio never leaves this computer: the browser sends it to this local server, it is
analysed in memory and discarded (nothing is written to disk).
"""
import io
import os
import sys
import json
import argparse
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                   # import config, data, audio_predictor
os.environ.setdefault("KERAS_BACKEND", "torch")            # same backend train.py uses

import librosa                                             # noqa: E402
import soundfile as sf                                     # noqa: E402
from flask import Flask, jsonify, request, send_from_directory   # noqa: E402

import config                                              # noqa: E402
import data                                                # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_SECONDS = 30            # longer uploads are cut to the first 30 s
WINDOW_S = config.DURATION  # the models were trained on 3 s clips
HOP_S = 1.5                 # 50 % overlap between analysis windows

# Preference order when picking the default model (best test accuracy first).
PREFERRED = ["wav2vec2_6class", "wav2vec2_extra", "wavlm_large_extra", "wav2vec2_extra_s7",
             "wav2vec2", "wavlm_large_full", "wavlm_large", "cnn_bilstm", "mlp", "cnn"]

# Test accuracy reported in results/ (shown in the app's "About" page).
ACCURACY = {}
for _name in os.listdir(config.RESULTS_DIR) if os.path.isdir(config.RESULTS_DIR) else []:
    if _name.endswith(".json"):
        try:
            with open(os.path.join(config.RESULTS_DIR, _name)) as _f:
                _r = json.load(_f)
            ACCURACY[_name[:-5]] = _r.get("test_accuracy", _r.get("accuracy"))
        except (OSError, ValueError):
            pass


# ----------------------------------------------------------------------------
# Model discovery + loading
# ----------------------------------------------------------------------------
def discover_models():
    """Every usable model folder in saved_models/: Hugging Face (config.json) or Keras (model.keras)."""
    found = {}
    if not os.path.isdir(config.MODELS_DIR):
        return found
    for name in os.listdir(config.MODELS_DIR):
        d = os.path.join(config.MODELS_DIR, name)
        if os.path.isfile(os.path.join(d, "config.json")) and (
                os.path.isfile(os.path.join(d, "model.safetensors")) or
                os.path.isfile(os.path.join(d, "pytorch_model.bin"))):
            with open(os.path.join(d, "config.json")) as f:
                id2label = json.load(f).get("id2label", {})
            labels = [id2label[k] for k in sorted(id2label, key=int)] or list(config.EMOTIONS)
            found[name] = {"kind": "transformer", "labels": labels}
        elif os.path.isfile(os.path.join(d, "model.keras")):
            labels = list(config.EMOTIONS)
            meta = os.path.join(d, "meta.json")
            if os.path.isfile(meta):
                with open(meta) as f:
                    labels = json.load(f).get("class_names", labels)
            found[name] = {"kind": "keras", "labels": labels}
    for name, info in found.items():
        acc = ACCURACY.get(name)
        info["accuracy"] = round(float(acc), 3) if acc is not None else None
    order = {n: i for i, n in enumerate(PREFERRED)}
    return dict(sorted(found.items(), key=lambda kv: order.get(kv[0], len(order))))


class Model:
    """Wraps one trained model: predict(window) -> probabilities over self.labels."""

    def __init__(self, name, info):
        self.name, self.labels, self.kind = name, info["labels"], info["kind"]
        d = os.path.join(config.MODELS_DIR, name)
        if self.kind == "transformer":
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.fe = AutoFeatureExtractor.from_pretrained(d)
            self.model = AutoModelForAudioClassification.from_pretrained(d).to(self.device).eval()
            self.sr = config.W2V_SR
        else:
            from audio_predictor import AudioEmotionPredictor   # reuses the exact training preprocessing
            self.pred = AudioEmotionPredictor(name)
            self.sr = self.pred.sr

    def predict(self, y):
        """y: mono waveform at self.sr (any length) -> probs, same pipeline as training."""
        if self.kind == "keras":
            return self.pred.predict_waveform(y, self.sr)
        y = data.prepare_waveform(y, sr=self.sr)
        inputs = self.fe(y, sampling_rate=self.sr, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(inputs["input_values"].to(self.device)).logits
        return self.torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)


_models, _lock = {}, threading.Lock()


def get_model(name, available):
    with _lock:
        if name not in _models:
            _models[name] = Model(name, available[name])
        return _models[name]


# ----------------------------------------------------------------------------
# Signal analysis
# ----------------------------------------------------------------------------
def voice_metrics(y, sr):
    """Simple, explainable measurements of HOW the person spoke."""
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    thresh = max(0.02 * rms.max(), 1e-4)
    voiced = rms > thresh
    f0 = librosa.yin(y, fmin=65, fmax=450, sr=sr, frame_length=2048, hop_length=hop)
    n = min(len(f0), len(voiced))
    f0v = f0[:n][voiced[:n]]
    f0v = f0v[(f0v > 70) & (f0v < 440)]
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="time")
    speech_s = voiced.sum() * hop / sr
    return {
        "pitch_hz": round(float(np.median(f0v)), 1) if f0v.size else None,
        # spread of pitch in semitones: low = flat/monotone, high = expressive
        "pitch_variation": round(float(np.std(12 * np.log2(f0v / np.median(f0v)))), 2) if f0v.size > 5 else None,
        "energy_db": round(float(20 * np.log10(np.mean(rms[voiced]) + 1e-9)), 1) if voiced.any() else None,
        "pause_ratio": round(float(1 - voiced.mean()), 3),
        "pace": round(float(len(onsets) / speech_s), 2) if speech_s > 0.3 else None,   # onsets per second of speech
        "duration_s": round(len(y) / sr, 2),
    }


def windows(y, sr):
    """Split speech into 3 s windows with 50 % overlap (one window if short)."""
    win, hop = int(WINDOW_S * sr), int(HOP_S * sr)
    if len(y) <= win + hop // 2:
        return [(0.0, y)]
    starts = list(range(0, len(y) - win + 1, hop))
    if len(y) - (starts[-1] + win) > hop // 2:
        starts.append(len(y) - win)
    return [(s / sr, y[s:s + win]) for s in starts]


# ----------------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


@app.get("/api/models")
def api_models():
    available = discover_models()
    return jsonify({"models": [{"name": n, **i} for n, i in available.items()],
                    "default": next(iter(available), None)})


@app.post("/api/analyze")
def api_analyze():
    available = discover_models()
    if not available:
        return jsonify(error="No trained model found in saved_models/. Train one first (see README)."), 503
    name = request.form.get("model") or next(iter(available))
    if name not in available:
        return jsonify(error=f"Unknown model '{name}'."), 400
    f = request.files.get("audio")
    if f is None:
        return jsonify(error="No audio received."), 400

    try:
        y, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=True)
    except Exception:
        return jsonify(error="Could not read that audio. Please record again or upload a .wav file."), 400
    y = y.mean(axis=1)[: int(MAX_SECONDS * sr)]

    if y.size < 0.5 * sr or np.max(np.abs(y)) < 0.01:
        return jsonify(error="We couldn't hear much. Try speaking a little closer to the microphone."), 422

    try:
        model = get_model(name, available)
    except Exception as e:                                    # missing library, corrupt files, ...
        return jsonify(error=f"Could not load model '{name}': {e}"), 500

    y = librosa.resample(y, orig_sr=sr, target_sr=model.sr) if sr != model.sr else y
    speech, _ = librosa.effects.trim(y, top_db=config.TOP_DB)
    if speech.size < 0.5 * model.sr:
        return jsonify(error="That was mostly silence. Try saying a sentence or two."), 422

    timeline, all_probs = [], []
    for start, w in windows(speech, model.sr):
        p = model.predict(w)
        all_probs.append(p)
        timeline.append({"start": round(start, 2), "emotion": model.labels[int(np.argmax(p))],
                         "confidence": round(float(p.max()), 3)})
    probs = np.mean(all_probs, axis=0)
    top = int(np.argmax(probs))

    return jsonify({
        "model": name,
        "model_accuracy": available[name]["accuracy"],
        "emotion": model.labels[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {lab: round(float(p), 4) for lab, p in zip(model.labels, probs)},
        "timeline": timeline,
        "metrics": voice_metrics(speech, model.sr),
    })


def main():
    ap = argparse.ArgumentParser(description="Feelspace - voice check-in web app")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    models = discover_models()
    print("Models found:", ", ".join(models) or "none (train one first)")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
