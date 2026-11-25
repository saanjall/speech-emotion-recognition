# Speech Emotion Recognition — Work Summary

**Task:** recognise 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech.
**Main dataset:** RAVDESS (24 actors, 1440 speech clips).
**Evaluation:** speaker-independent — train actors 1–18, validation 19–20, **test 21–24** (240 clips the
models never heard). Chance level = 12.5 %.

## Final results (RAVDESS test actors 21–24)

| Rank | Model | Test accuracy | Macro-F1 | Val acc | Training data |
|---|---|---|---|---|---|
| 1 | **Ensemble, equal average of #2 + #4** | **80.0 %** | **0.792** | 79.2 % | — (nothing tuned) |
| 2 | wav2vec2-base + CREMA-D/TESS | 79.2 % | 0.784 | 74.2 % | RAVDESS + CREMA-D + TESS |
| 3 | wav2vec2-base + CREMA-D/TESS (seed 7) | 77.1 % | 0.767 | 71.7 % | RAVDESS + CREMA-D + TESS |
| 4 | WavLM-large + ½ CREMA-D | 76.7 % | 0.757 | 81.7 % | RAVDESS + 50 % of CREMA-D |
| 5 | wav2vec2-base | 74.6 % | 0.739 | 78.3 % | RAVDESS |
| 6 | Ensemble (weights tuned on validation) | 74.2 % | 0.727 | 85.0 % | — (all models) |
| 7 | WavLM-large + CREMA-D/TESS | 73.8 % | 0.719 | 84.2 % | RAVDESS + CREMA-D + TESS |
| 8 | WavLM-large | 72.5 % | 0.707 | 75.8 % | RAVDESS |
| 9 | CNN + BiLSTM + Attention | 59.6 % | 0.572 | 63.3 % | RAVDESS |
| 10 | MLP (handcrafted features) | 50.0 % | 0.489 | 52.5 % | RAVDESS |
| 11 | CNN (log-mel) | 45.4 % | 0.445 | 60.8 % | RAVDESS |

An equal average of all four CREMA-D models (#2, #3, #4, #7) gave 79.2 % — no better than the 2-model average.

### Other evaluations (6 emotions = CREMA-D's set: neutral, happy, sad, angry, fearful, disgust)

| Model | Test set | Test accuracy | Macro-F1 | Test clips |
|---|---|---|---|---|
| **wav2vec2-base, trained on 6 emotions** (RAVDESS + CREMA-D + TESS) | RAVDESS actors 21–24 | **86.4 %** | **0.860** | 176 |
| wav2vec2-base, CREMA-D only | 18 unseen CREMA-D actors | 70.0 % | 0.701 | 1,464 |

These are easier / different tasks than the 8-emotion benchmark above and are reported **alongside** it,
not instead of it. See step 9.

Chart: `results/comparison.png` · table: `results/comparison.md` · per model: `results/<model>_confusion.png`,
`results/<model>_curves.png`, `results/<model>.json`.

**Per emotion** (best model = 2-model equal average, recall on test):

| Strong | Recall | Weak | Recall | Usually confused with |
|---|---|---|---|---|
| surprised | 100 % | happy | 59 % | surprised |
| neutral | 94 % | sad | 59 % | calm |
| calm | 94 % | fearful | 59 % | sad |
| disgust | 94 % | | | |
| angry | 88 % | | | |

The weak emotions sound like a neighbour (happy/surprised are both high-energy, sad/calm both quiet and
slow), and the models over-predict calm/surprised/neutral (precision 0.65–0.73) when unsure. Fearful has
0.90 precision but 0.59 recall: when the model says "fearful" it is right, but it rarely says it.
CREMA-D (most of the training data) has no calm or surprised clips, so those pairs are learned from
RAVDESS only.

## What was done, step by step

### 1. Environment
- Python 3.12 virtual environment, all `requirements.txt` + `requirements-optional.txt` packages.
- PyTorch 2.11 with CUDA 12.8 — runs on the laptop's **NVIDIA RTX 4060 (8 GB)**.
- TensorFlow has no GPU support on native Windows, so the Keras models were switched to Keras 3's
  **PyTorch backend** (see step 3).

### 2. Data
- **RAVDESS** speech (official Zenodo copy): 1440 clips, split strictly by actor.
- `data.py` extracts features once (log-mel 128×130 for the CNNs, 222-d MFCC/chroma/mel/ZCR/RMS
  vector for the MLP) with augmentation (noise, pitch shift, time stretch, time shift, SpecAugment)
  **on the training actors only** → no leakage into validation/test.

### 3. Speed-up: Keras models on the GPU
- `models.py`: the custom attention layer now uses `keras.ops` instead of TensorFlow-only calls.
- `train.py`: uses the PyTorch backend (`KERAS_BACKEND=torch`).
- Result: CNN epoch 42 s → 8 s, CNN+BiLSTM epoch 116 s → 17 s (**~5–7× faster**); all three Keras
  models now train in ~16 minutes instead of ~1.5–3 hours. Trained models still load under TensorFlow.

### 4. Baseline models (from scratch)
MLP, CNN and CNN+BiLSTM+Attention trained on RAVDESS: 50.0 %, 45.4 %, 59.6 %.
These models start with no knowledge of speech, so with only 18 training speakers they overfit
(e.g. CNN: ~92 % train vs ~50 % validation accuracy).

### 5. Pretrained speech models (transfer learning)
- **wav2vec2-base** (pretrained on 960 h of speech), fine-tuned on RAVDESS: first run 79.2 %,
  a later re-run with 30 epochs + weighted layer sum 74.6 % (the run kept in `results/`).
  The difference shows how much single runs vary on a 240-clip test set.
- **WavLM-large** (316 M parameters): `train_wav2vec2.py` was generalised to any Hugging Face
  audio model (`AutoModelForAudioClassification`), batch size 4 to fit in 8 GB → 72.5 %.

### 6. Ensemble
`ensemble.py` averages the models' probabilities with weights chosen **on the validation actors
only**. It reached 85 % on validation but 74.2 % on test — the 120-clip validation set is too small
to tune weights reliably (an honest negative result).

### 7. More speakers: CREMA-D + TESS
Diagnosis: every model overfits to the 18 RAVDESS training voices. So two public corpora were added
to the **training set only**:
- **CREMA-D** — 91 actors, 7442 clips (6 emotions; no calm/surprised).
- **TESS** — 2 speakers, 2800 clips (7 emotions; "pleasant surprise" → surprised).

Training set: 11,322 clips from 111 speakers. Validation and test are **unchanged RAVDESS actors
19–24**, so the result is directly comparable with all other rows → **79.2 %, macro-F1 0.784**
(best single model). Remaining errors mostly involve calm/surprised, which the extra corpora do not contain.

### 8. Bigger model, more runs, equal-weight ensemble (best result)
Attempt to reach 85 %:
- **WavLM-large + ½ CREMA-D** (TESS dropped, half of CREMA-D kept with all 91 speakers, for speed:
  4,801 clips, ~5 min/epoch) → 76.7 % test, 81.7 % val.
- **WavLM-large + full CREMA-D/TESS** (11,322 clips, ~12 min/epoch) → 73.8 % test despite the best
  validation score (84.2 %); sad and happy recall collapsed to 34 % / 38 %.
- **Second wav2vec2-base + CREMA-D/TESS run (seed 7)** → 77.1 %.
- **Equal-weight average** (`ensemble.py --equal ...`, no weights fitted on validation):
  wav2vec2-base + CREMA-D/TESS with WavLM-large + ½ CREMA-D → **80.0 %, macro-F1 0.792** (best result).
  Averaging all four CREMA-D models gave 79.2 %. Which models go into the average was fixed before
  looking at the test score.

Finding: across these runs, higher validation accuracy went with *lower* test accuracy — two
validation actors cannot tell which change genuinely helps, so results plateau around 79–80 %.

### 9. Six emotions and a CREMA-D test set
CREMA-D (most of the training data) has no calm or surprised, and those two classes caused many of
the 8-emotion errors (happy → surprised, sad → calm). So the emotion set shared by both datasets was
used: neutral, happy, sad, angry, fearful, disgust (`train_wav2vec2.py --six-emotions`).
- **RAVDESS test, 6 emotions:** train RAVDESS 1–18 + CREMA-D + TESS without calm/surprised clips
  (10,634 clips), val RAVDESS 19–20 (88), test RAVDESS 21–24 (176). Same settings as the 79.2 % model.
  → **86.4 %, macro-F1 0.860.** Recall: neutral 100 %, disgust 100 %, angry 97 %, happy 84 %,
  fearful 75 %, sad 69 % (sad → disgust is the main error). For comparison, the 8-emotion model
  scored 79.0 % on the same 176 clips when its predictions were restricted to these 6 emotions.
- **CREMA-D test** (`--crema-only`): CREMA-D alone, split by actor with a fixed seed — 64 train /
  9 val / 18 test actors (5,240 / 738 / 1,464 clips). → **70.0 %, macro-F1 0.701.** CREMA-D is harder
  (subtler acting, 12 sentences; human audio-only agreement with the intended emotion ≈ 41 %), but
  the test is 8× larger, so the number is more reliable (≈ ±2.3 points vs ≈ ±5.5 for RAVDESS).
  Recall: angry 86 %, happy 81 %, neutral 77 %, disgust 61 %, fearful 59 %, sad 57 %.

Sad and fearful are the weakest emotions on both datasets, so this is a genuine model limitation.

## Why the evaluation is trustworthy
- **Speaker-independent split** — no test actor's voice is ever used for training or model selection.
- **Augmentation, normalisation statistics and class weights use training data only.**
- **Test set used once per model**; all choices (epochs, ensemble weights) were made on validation.
- We deliberately **did not** use a random split, tune on the test set, or use public "RAVDESS emotion"
  checkpoints (many were trained on splits that include our test actors) — all of these would inflate
  the score. Random-split tutorials report 90 %+, but that measures voice recognition, not emotion.

## Limitations and next steps
- 240 test clips → ±3–5 points run-to-run variation; 79.2 % vs 80 % is 2 clips.
- Validation set (120 clips, 2 actors) is noisy for early stopping / ensemble weighting.
- Happy, sad and fearful (59 % recall each) are the gap to 85 %.
- Next: 6-fold actor cross-validation (every actor tested once; ~3–4 h with wav2vec2-base) for a
  more reliable number, hold out some CREMA-D actors as a second test set, or add a corpus that
  contains "calm" and "surprised".

## Code changes (see `code_changes/`)
| File | Change |
|---|---|
| `models.py` | attention layer uses `keras.ops` (works on TensorFlow and PyTorch backends) |
| `train.py` | Keras trains on the PyTorch backend → GPU on Windows |
| `train_wav2vec2.py` | any HF audio model (wav2vec2 / WavLM), extra training corpora, parallel audio loading; progress bar; `--model --run --corpora --fraction --seed` override config per run; `--six-emotions`, `--crema-only` |
| `config.py` | WavLM / extra-data settings (`W2V_EXTRA_FRACTION`), run names, new model names |
| `data.py` | `list_extra_files()` — CREMA-D + TESS loaders with label mapping |
| `audio_predictor.py` | loads any HF audio model; fixes GPU-tensor → NumPy on the PyTorch backend |
| `compare.py` | new models in the table, readable chart labels |
| `ensemble.py` | **new** — validation-weighted ensemble; `--equal` for a plain average of chosen models |

## How to reproduce
```
C:\venvs\ser\Scripts\Activate.ps1
python data.py                 # features (once)
python train.py                # MLP, CNN, CNN+BiLSTM (GPU via PyTorch backend)
python train_wav2vec2.py       # model/data chosen in config.py (W2V_MODEL_NAME, W2V_EXTRA_CORPORA)
python train_wav2vec2.py --model facebook/wav2vec2-base --run wav2vec2_extra --corpora CREMA-D TESS --fraction 1 --batch 8 --lr 5e-5
python train_wav2vec2.py --model microsoft/wavlm-large --run wavlm_large_extra --corpora CREMA-D --fraction 0.5 --batch 4 --lr 3e-5
python ensemble.py --equal wav2vec2_extra wavlm_large_extra   # best result (80.0 %)
python train_wav2vec2.py --six-emotions --model facebook/wav2vec2-base --run wav2vec2_6class --corpora CREMA-D TESS --fraction 1 --batch 8 --lr 5e-5   # 86.4 %, 6 emotions
python train_wav2vec2.py --crema-only --model facebook/wav2vec2-base --run wav2vec2_crema_only --batch 8 --lr 5e-5   # 70.0 %, CREMA-D test
python ensemble.py             # optional: validation-weighted ensemble
python compare.py              # table + chart
```
Datasets go in `data/RAVDESS`, `data/CREMA-D/AudioWAV`, `data/TESS` (not included — too large).
