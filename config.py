"""
config.py - every tunable setting of the project lives here.

Keeping all constants in one file means the viva answer to
"where do you change X?" is always: config.py.
"""
import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "RAVDESS")          # Actor_01 .. Actor_24
FER_DIR = os.path.join(ROOT_DIR, "data", "fer2013")           # train/<emotion>/, test/<emotion>/
CREMA_DIR = os.path.join(ROOT_DIR, "data", "CREMA-D")          # AudioWAV/1001_DFA_ANG_XX.wav  (extra TRAIN data)
TESS_DIR = os.path.join(ROOT_DIR, "data", "TESS")              # OAF_back_angry.wav            (extra TRAIN data)
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "features.npz")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# --------------------------------------------------------------------------
# Labels (RAVDESS 3rd filename field, 01..08 -> index 0..7)
# --------------------------------------------------------------------------
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_CODES = {f"{i + 1:02d}": name for i, name in enumerate(EMOTIONS)}
NUM_CLASSES = len(EMOTIONS)

# Only audio-only (03) speech (01) files are used.
MODALITY_AUDIO_ONLY = "03"
VOCAL_CHANNEL_SPEECH = "01"

# --------------------------------------------------------------------------
# Speaker-independent split (by actor id).  No actor appears in two splits.
# --------------------------------------------------------------------------
TRAIN_ACTORS = list(range(1, 19))    # 1..18
VAL_ACTORS = [19, 20]
TEST_ACTORS = [21, 22, 23, 24]

# --------------------------------------------------------------------------
# Audio preprocessing
# --------------------------------------------------------------------------
SR = 22050              # sample rate for the Keras models
DURATION = 3.0          # seconds; every clip is centre-cropped / zero-padded to this
TOP_DB = 30             # librosa.effects.trim threshold
N_SAMPLES = int(SR * DURATION)

# Log-mel spectrogram (CNN inputs)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH     # = 130 frames for 3 s @ 22.05 kHz

# Handcrafted vector (MLP input)
N_MFCC = 40

# --------------------------------------------------------------------------
# Augmentation (TRAIN SET ONLY)
# --------------------------------------------------------------------------
N_AUGMENT = 2                    # extra augmented copies per training clip
NOISE_FACTOR = (0.002, 0.01)     # Gaussian noise std relative to peak amplitude 1.0
PITCH_STEPS = (-2.0, 2.0)        # semitones
STRETCH_RATE = (0.85, 1.15)
SHIFT_MAX = 0.2                  # max time shift as fraction of the clip
FREQ_MASK_MAX = 15               # SpecAugment: max mel bins masked
TIME_MASK_MAX = 20               # SpecAugment: max frames masked

# --------------------------------------------------------------------------
# Training (Keras)
# --------------------------------------------------------------------------
SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 5
N_JOBS = -1                      # joblib workers for feature extraction (-1 = all cores)

# --------------------------------------------------------------------------
# wav2vec2 (optional, PyTorch)
# --------------------------------------------------------------------------
# Runs so far:  "facebook/wav2vec2-base" -> "wav2vec2" (20 ep, batch 8, LR 5e-5)
#               "microsoft/wavlm-large"  -> "wavlm_large" (20 ep, batch 4, LR 3e-5)
#               "facebook/wav2vec2-base" + CREMA-D/TESS -> "wav2vec2_extra" (10 ep, batch 8, LR 5e-5)
W2V_MODEL_NAME = "microsoft/wavlm-large"
W2V_RUN_NAME = "wavlm_large_extra"  # saved_models/<name>/ and results/<name>.json
W2V_EXTRA_CORPORA = ["CREMA-D"]   # added to the TRAINING set only (val/test stay RAVDESS); TESS dropped for speed (only 2 speakers)
W2V_EXTRA_FRACTION = 0.5          # random share of the extra clips to use (1 = all); 0.5 of CREMA-D ~ 3.7k clips, all 91 speakers
W2V_SR = 16000
W2V_EPOCHS = 10                  # ~10x more clips per epoch than RAVDESS alone (TESS only: ~3.6x -> use 15)
W2V_BATCH = 4                    # wavlm-large needs 4 to fit in 8 GB
W2V_LR = 3e-5
W2V_PATIENCE = 4

# --------------------------------------------------------------------------
# Face emotion + fusion
# --------------------------------------------------------------------------
FACE_IMG_SIZE = 48
FACE_EPOCHS = 60
FACE_BATCH = 64
FACE_N_FRAMES = 6                # frames analysed during a live recording
AUDIO_WEIGHT = 0.7               # fused = w * audio + (1 - w) * face
MIN_AUDIO_WEIGHT = 0.5           # audio must always dominate

# FER-2013 (7 classes) -> RAVDESS (8 classes).  "neutral" is split 50/50.
FACE_TO_AUDIO = {
    "angry": {"angry": 1.0},
    "disgust": {"disgust": 1.0},
    "fear": {"fearful": 1.0},
    "happy": {"happy": 1.0},
    "sad": {"sad": 1.0},
    "surprise": {"surprised": 1.0},
    "neutral": {"neutral": 0.5, "calm": 0.5},
}

TORCH_AUDIO_MODELS = ["wav2vec2", "wavlm_large", "wav2vec2_tess", "wav2vec2_extra", "wavlm_large_extra",
                      "wavlm_large_full", "wav2vec2_extra_s7"]   # trained by train_wav2vec2.py
AUDIO_MODELS = ["mlp", "cnn", "cnn_bilstm"] + TORCH_AUDIO_MODELS
