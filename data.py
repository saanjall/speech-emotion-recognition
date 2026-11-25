"""
data.py - RAVDESS file discovery, preprocessing, augmentation, features, caching.

Pipeline for ONE clip:
    load (22.05 kHz mono) -> prepare_waveform (trim, normalise, crop/pad 3 s)
        -> [train only] augment waveform (+ SpecAugment on its spectrogram)
        -> features: 2-D log-mel (CNNs) and 1-D handcrafted vector (MLP)

Golden rule: we SPLIT BY ACTOR FIRST, then augment only the training clips.
Validation / test clips are never augmented, so nothing "leaks" between splits.
"""
import os
import json
import argparse

import numpy as np
import librosa
from joblib import Parallel, delayed

import config


# ==========================================================================
# 1. File discovery
# ==========================================================================
def parse_filename(filename):
    """RAVDESS name: modality-channel-emotion-intensity-statement-repetition-actor.wav
    e.g. 03-01-05-01-02-01-12.wav -> audio-only, speech, angry, ..., actor 12."""
    parts = os.path.splitext(os.path.basename(filename))[0].split("-")
    if len(parts) != 7:
        return None
    return {
        "modality": parts[0],
        "channel": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "actor": int(parts[6]),
    }


def list_ravdess_files(data_dir=config.DATA_DIR):
    """Walk data_dir recursively and return a list of (path, label_idx, actor).

    The Kaggle zip contains every file twice (Actor_xx/ and
    audio_speech_actors_01-24/Actor_xx/), so we de-duplicate by FILENAME.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"RAVDESS folder not found: {data_dir} (see README, 'Dataset').")

    seen, items = set(), []
    for root, _, files in os.walk(data_dir):
        for name in sorted(files):
            if not name.lower().endswith(".wav") or name in seen:
                continue
            info = parse_filename(name)
            if info is None:
                continue
            if info["modality"] != config.MODALITY_AUDIO_ONLY or info["channel"] != config.VOCAL_CHANNEL_SPEECH:
                continue
            if info["emotion"] not in config.EMOTION_CODES:
                continue
            seen.add(name)
            label = int(info["emotion"]) - 1
            items.append((os.path.join(root, name), label, info["actor"]))
    items.sort(key=lambda t: os.path.basename(t[0]))
    return items


def split_by_actor(items):
    """Speaker-independent split -> dict of lists. No actor is in two splits."""
    splits = {"train": [], "val": [], "test": []}
    for item in items:
        actor = item[2]
        if actor in config.TRAIN_ACTORS:
            splits["train"].append(item)
        elif actor in config.VAL_ACTORS:
            splits["val"].append(item)
        elif actor in config.TEST_ACTORS:
            splits["test"].append(item)
    return splits


# ==========================================================================
# 2. Waveform preprocessing (shared by training AND live microphone input)
# ==========================================================================
def fix_length(y, n_samples):
    """Centre-crop a long signal or symmetrically zero-pad a short one."""
    if len(y) > n_samples:
        start = (len(y) - n_samples) // 2
        return y[start:start + n_samples]
    pad = n_samples - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2))


def prepare_waveform(y, sr=config.SR, duration=config.DURATION, top_db=config.TOP_DB):
    """Trim leading/trailing silence, peak-normalise, then crop/pad to `duration` seconds.

    The SAME function is used for dataset files and for live mic recordings,
    so the model always sees inputs prepared in exactly the same way.
    """
    y = np.asarray(y, dtype=np.float32).flatten()
    if y.size == 0:
        return np.zeros(int(sr * duration), dtype=np.float32)
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size > 0.1 * sr:          # keep the original if trimming removed almost everything
        y = trimmed
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak
    return fix_length(y, int(sr * duration)).astype(np.float32)


def load_audio(path, sr=config.SR):
    """Load a file as mono float32 at the requested sample rate and prepare it."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return prepare_waveform(y, sr)


# ==========================================================================
# 3. Augmentation (TRAIN ONLY)
# ==========================================================================
def add_noise(y, rng, sr=None):
    return y + rng.uniform(*config.NOISE_FACTOR) * rng.standard_normal(len(y)).astype(np.float32)


def pitch_shift(y, rng, sr=config.SR):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=rng.uniform(*config.PITCH_STEPS))


def time_stretch(y, rng, sr=None):
    return librosa.effects.time_stretch(y, rate=rng.uniform(*config.STRETCH_RATE))


def time_shift(y, rng, sr=None):
    max_shift = int(config.SHIFT_MAX * len(y))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    out = np.roll(y, shift)
    if shift > 0:
        out[:shift] = 0.0          # zero the wrapped-around part instead of looping audio
    elif shift < 0:
        out[shift:] = 0.0
    return out


AUGMENTATIONS = [add_noise, pitch_shift, time_stretch, time_shift]


def augment_waveform(y, rng, sr=config.SR):
    """Apply 1-2 randomly chosen augmentations, then restore the fixed length."""
    n_ops = int(rng.integers(1, 3))
    for idx in rng.choice(len(AUGMENTATIONS), size=n_ops, replace=False):
        y = AUGMENTATIONS[idx](y, rng, sr=sr)
    y = fix_length(np.asarray(y, dtype=np.float32), int(sr * config.DURATION))   # stretch changes length
    peak = np.max(np.abs(y))
    return (y / peak if peak > 1e-6 else y).astype(np.float32)


def spec_augment(mel, rng):
    """SpecAugment: zero one random band of mel bins and one random block of frames.
    The mask value is the spectrogram minimum (= 'silence' in log scale)."""
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    fill = mel.min()
    f = int(rng.integers(0, config.FREQ_MASK_MAX + 1))
    f0 = int(rng.integers(0, max(1, n_mels - f)))
    mel[f0:f0 + f, :] = fill
    t = int(rng.integers(0, config.TIME_MASK_MAX + 1))
    t0 = int(rng.integers(0, max(1, n_frames - t)))
    mel[:, t0:t0 + t] = fill
    return mel


# ==========================================================================
# 4. Feature extraction
# ==========================================================================
def extract_logmel(y, sr=config.SR):
    """2-D log-mel spectrogram, shape (N_MELS, N_FRAMES) = (128, 130)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=config.N_FFT,
                                         hop_length=config.HOP_LENGTH, n_mels=config.N_MELS)
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
    # guarantee a fixed number of frames
    if logmel.shape[1] < config.N_FRAMES:
        logmel = np.pad(logmel, ((0, 0), (0, config.N_FRAMES - logmel.shape[1])), constant_values=logmel.min())
    return logmel[:, :config.N_FRAMES].astype(np.float32)


def extract_handcrafted(y, sr=config.SR):
    """1-D vector for the MLP:
    MFCC(40) mean + std (80) | chroma mean (12) | mel mean (128) | ZCR mean (1) | RMS mean (1) = 222 dims."""
    stft = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(S=stft ** 2, sr=sr)
    mel = librosa.feature.melspectrogram(S=stft ** 2, sr=sr, n_mels=config.N_MELS)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=config.HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=config.HOP_LENGTH)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1),
        librosa.power_to_db(mel.mean(axis=1)),
        zcr.mean(axis=1), rms.mean(axis=1),
    ]).astype(np.float32)


HANDCRAFTED_DIM = 2 * config.N_MFCC + 12 + config.N_MELS + 2


# ==========================================================================
# 5. Dataset building + caching
# ==========================================================================
def _process_file(path, label, actor, split_id, n_aug, seed):
    """Worker run in parallel: returns the original clip + n_aug augmented copies."""
    rng = np.random.default_rng(seed)
    y = load_audio(path)
    out = [(extract_logmel(y), extract_handcrafted(y), label, actor, split_id, 0)]
    for _ in range(n_aug):
        y_aug = augment_waveform(y, rng)
        mel = spec_augment(extract_logmel(y_aug), rng)          # SpecAugment on augmented copies only
        out.append((mel, extract_handcrafted(y_aug), label, actor, split_id, 1))
    return out


def build_dataset(data_dir=config.DATA_DIR, cache_path=config.CACHE_PATH, rebuild=False, n_jobs=config.N_JOBS):
    """Extract features for all files (in parallel) and cache them to an .npz file.

    Returns a dict with arrays: mel, vec, y, actor, split (0 train / 1 val / 2 test), aug (0/1).
    """
    if os.path.exists(cache_path) and not rebuild:
        print(f"[data] loading cached features from {cache_path}")
        return dict(np.load(cache_path))

    items = list_ravdess_files(data_dir)
    if not items:
        raise RuntimeError(f"No RAVDESS speech files found in {data_dir}")
    splits = split_by_actor(items)                     # SPLIT FIRST ...
    print("[data] files per split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    jobs = []
    for split_id, name in enumerate(["train", "val", "test"]):
        n_aug = config.N_AUGMENT if name == "train" else 0     # ... THEN augment train only
        for i, (path, label, actor) in enumerate(splits[name]):
            jobs.append(delayed(_process_file)(path, label, actor, split_id, n_aug,
                                               config.SEED + 100000 * split_id + i))

    print(f"[data] extracting features for {len(jobs)} files with joblib (n_jobs={n_jobs}) ...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(jobs)
    rows = [row for file_rows in results for row in file_rows]

    data = {
        "mel": np.stack([r[0] for r in rows]),
        "vec": np.stack([r[1] for r in rows]),
        "y": np.array([r[2] for r in rows], dtype=np.int64),
        "actor": np.array([r[3] for r in rows], dtype=np.int64),
        "split": np.array([r[4] for r in rows], dtype=np.int64),
        "aug": np.array([r[5] for r in rows], dtype=np.int64),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f"[data] cached {len(rows)} samples -> {cache_path}")
    return data


def get_split(data, name, feature):
    """Return (X, y) for split name in {'train','val','test'} and feature in {'mel','vec'}."""
    mask = data["split"] == {"train": 0, "val": 1, "test": 2}[name]
    return data[feature][mask], data["y"][mask]


# ==========================================================================
# 6. Normalisation (statistics computed on TRAIN ONLY, then saved)
# ==========================================================================
class SpecNormalizer:
    """Per-mel-bin mean/std standardisation for spectrograms, saved as JSON."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def fit(self, mels):                        # mels: (N, n_mels, n_frames)
        self.mean = mels.mean(axis=(0, 2)).astype(np.float32)
        self.std = (mels.std(axis=(0, 2)) + 1e-6).astype(np.float32)
        return self

    def transform(self, mels):
        return (mels - self.mean[None, :, None]) / self.std[None, :, None]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["mean"], d["std"])


# ==========================================================================
# 7. Extra TRAINING corpora: CREMA-D (91 actors) + TESS (2 speakers)
#    More speakers -> better on unseen voices. They are ONLY added to the training
#    set; validation/test stay RAVDESS actors 19-24, so results remain comparable.
# ==========================================================================
CREMA_CODES = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}
TESS_CODES = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
              "fear": "fearful", "disgust": "disgust", "ps": "surprised"}      # ps = pleasant surprise


def list_extra_files(corpora=("CREMA-D", "TESS"), crema_dir=config.CREMA_DIR, tess_dir=config.TESS_DIR):
    """(path, label_idx, speaker_id) for CREMA-D (1001_DFA_ANG_XX.wav) and TESS (OAF_back_angry.wav).
    Speaker ids: CREMA-D 1001-1091, TESS 2001 (OAF) / 2002 (YAF) - never clash with RAVDESS 1-24."""
    items = []
    for root, _, files in os.walk(crema_dir) if "CREMA-D" in corpora else []:
        for name in files:
            parts = os.path.splitext(name)[0].split("_")
            path = os.path.join(root, name)
            if name.lower().endswith(".wav") and len(parts) == 4 and parts[2] in CREMA_CODES \
                    and os.path.getsize(path) > 1000:                       # a few CREMA-D files are empty
                items.append((path, config.EMOTIONS.index(CREMA_CODES[parts[2]]), int(parts[0])))
    for root, _, files in os.walk(tess_dir) if "TESS" in corpora else []:
        for name in files:
            parts = os.path.splitext(name)[0].split("_")
            if name.lower().endswith(".wav") and len(parts) >= 3 and parts[-1].lower() in TESS_CODES:
                items.append((os.path.join(root, name), config.EMOTIONS.index(TESS_CODES[parts[-1].lower()]),
                              2001 if parts[0].upper() == "OAF" else 2002))
    return sorted(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build / inspect the cached feature file.")
    parser.add_argument("--rebuild", action="store_true", help="re-extract features even if the cache exists")
    args = parser.parse_args()
    d = build_dataset(rebuild=args.rebuild)
    for split_id, name in enumerate(["train", "val", "test"]):
        m = d["split"] == split_id
        print(f"{name:5s}: {m.sum():5d} samples ({d['aug'][m].sum()} augmented), actors {sorted(set(d['actor'][m]))}")
    print("mel shape", d["mel"].shape, "| vec shape", d["vec"].shape)
