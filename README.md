# Speech Emotion Recognition with optional Face Emotion fusion (audio-dominant)

Recognise 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech
using the RAVDESS dataset. Several models are trained on the **same speaker-independent split** and
compared — from a classic MLP baseline up to a CNN + BiLSTM + attention network and a fine-tuned
wav2vec2 transformer. A face-emotion model can optionally *support* the audio prediction through
weighted late fusion, but audio always carries at least 50 % of the decision.

## Pipeline

```
 Audio ──► preprocessing ──────► features ─────────────► audio model ──► 8 probs ─┐
 (mic / .wav)  trim silence,       log-mel 128x130 (CNNs)   MLP / CNN /             │ w_audio (>= 0.5)
               peak-normalise,     MFCC/chroma/mel/ZCR/RMS  CNN+BiLSTM+Attn /       │
               crop/pad to 3 s     vector, 222-d (MLP)      wav2vec2                ├──► weighted ──► emotion
                                                                                    │     fusion
 Webcam ──► face detect ─────────► face CNN ─────────────► 7 probs → map to 8 ─────┘ 1 - w_audio
 (optional)  Haar / DeepFace        FER-2013 CNN / DeepFace   (neutral → 50% neutral + 50% calm)

 fused = w_audio * audio_probs + (1 - w_audio) * face_probs      (no face found → audio-only)
```

## Project files

| File | Purpose |
|---|---|
| `config.py` | every setting: paths, actor split, audio/feature params, training hyper-parameters, fusion weight |
| `data.py` | RAVDESS file listing + de-duplication, `prepare_waveform()`, augmentation, SpecAugment, features, parallel extraction + cache |
| `models.py` | Keras MLP, CNN, CNN+BiLSTM+attention (custom `AttentionPooling` layer) |
| `metrics_utils.py` | `save_results()`: accuracy, macro-F1, report, confusion matrix PNG, learning curves PNG, JSON |
| `train.py` | train/evaluate the Keras models |
| `train_wav2vec2.py` | optional: fine-tune `facebook/wav2vec2-base` (PyTorch, Colab GPU) |
| `compare.py` | comparison table + bar chart from `results/*.json` |
| `audio_predictor.py` | `AudioEmotionPredictor` — loads any trained audio model, returns 8 probabilities |
| `face_emotion.py` | `FaceEmotionPredictor` (deepface / custom / auto) + 7→8 emotion mapping |
| `train_face_cnn.py` | optional: train our own face CNN on FER-2013 |
| `fusion.py` | audio-dominant late fusion + result table |
| `live_demo.py` | microphone (+ webcam) real-time demo |
| `predict.py` | predict from an audio file (+ optional face image) |
| `smoke_test.py` | checks everything builds/works on random dummy data (no dataset needed) |

## 1. Setup

Use **Python 3.10–3.12** (TensorFlow does not ship wheels for every newer Python).

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
```
```bash
pip install -r requirements.txt
```
Optional (DeepFace face backend, wav2vec2):
```bash
pip install -r requirements-optional.txt
```

## 2. Dataset

**RAVDESS speech** — Kaggle: `uwrfkaggler/ravdess-emotional-speech-audio`

1. Download the zip from Kaggle (or `kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio`).
2. Extract so that the actor folders are inside `data/RAVDESS/`:
   ```
   data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav
   ...
   data/RAVDESS/Actor_24/...
   ```
   The Kaggle zip also contains a duplicate `audio_speech_actors_01-24/Actor_xx` copy — you can leave it;
   files are **de-duplicated by filename** automatically (1440 unique clips).
3. Only audio-only (modality `03`) speech (channel `01`) files are used. Emotion = 3rd filename field.

**FER-2013 (optional, only for the custom face model)** — Kaggle: `msambare/fer2013`, extract to
`data/fer2013/train/<emotion>/` and `data/fer2013/test/<emotion>/`.

## 3. Commands, in order

```bash
python smoke_test.py                       # sanity check, no dataset needed (~1-2 min)
```
```bash
python data.py                             # extract + cache features (cache/features.npz); add --rebuild to redo
```
```bash
python train.py --model all                # MLP, CNN, CNN+BiLSTM  (or --model cnn_bilstm --epochs 80 --batch 32)
```
```bash
python train_wav2vec2.py                   # optional, GPU recommended (see Colab section)
```
```bash
python compare.py                          # table + results/comparison.png
```
```bash
python train_face_cnn.py                   # optional, only if you want the "custom" face backend
```
```bash
python predict.py --audio data/RAVDESS/Actor_21/03-01-05-01-01-01-21.wav
```
```bash
python predict.py --audio clip.wav --image face.jpg --face-backend auto
```
```bash
python live_demo.py --audio-model cnn_bilstm
```
```bash
python live_demo.py --face --face-backend auto --seconds 4 --audio-weight 0.7
```

Outputs: trained models in `saved_models/<model>/`, metrics/plots in `results/`.

## 4. Why the evaluation is designed this way

**Speaker-independent split (train actors 1–18, val 19–20, test 21–24).**
RAVDESS has 24 actors each saying the same 2 sentences many times. With a random split, clips of the
*same actor* land in both train and test, and the model can partly recognise the *voice* rather than the
*emotion* — accuracy looks great (often 90 %+) but drops sharply on a new person. Splitting by actor means
the test score measures performance on **unseen speakers**, which is what a real application faces.
Numbers are therefore lower than many online tutorials, but they are honest.

**Split first, augment only the training set.** If we augmented first and then split, a noisy/pitch-shifted
copy of a test clip could be in the training set — the model would have effectively seen the test clip
(data leakage). Augmentation is a training-time regulariser; validation and test must stay real, untouched
recordings. For the same reason the StandardScaler / spectrogram mean-std are computed on **train only**.

**Validation vs test.** Early stopping, LR scheduling and choosing the best epoch use the validation actors.
The test actors are used exactly once, for the final numbers.

## 5. Models

| Model | Input | Idea |
|---|---|---|
| MLP (baseline) | 222-d vector: MFCC(40) mean+std, chroma, mel mean, ZCR, RMS | classic tutorial approach, but averaging over time throws away temporal info |
| CNN | log-mel spectrogram 128×130 | treats the spectrogram like an image; 4 conv blocks (32/64/128/256) → global average pooling |
| **CNN + BiLSTM + Attention** (main) | log-mel 128×130 | conv blocks pool only along frequency, so the 130 time steps survive → BiLSTM models how emotion evolves over time → attention picks the most emotional frames |
| wav2vec2-base (optional) | raw 16 kHz waveform | transformer pre-trained on 960 h of speech, fine-tuned (feature encoder frozen) |

Training: Adam (1e-3), sparse categorical cross-entropy, balanced class weights (neutral has half the clips),
EarlyStopping on val accuracy (patience 15, best weights restored), ReduceLROnPlateau.
Augmentation (train only, 2 extra copies/clip, 1–2 random ops): Gaussian noise, pitch shift ±2 st,
time stretch 0.85–1.15, time shift, plus SpecAugment (1 frequency + 1 time mask) on augmented spectrograms.

## 6. Results (`python compare.py` prints this table; test = RAVDESS actors 21–24)

| Rank | Model | Test accuracy | Macro-F1 | Best val acc | Params |
|---|---|---|---|---|---|
| 1 | **Ensemble (equal avg of #2 + #4)** | **0.800** | **0.792** | 0.792 | – |
| 2 | wav2vec2-base + CREMA-D/TESS | 0.792 | 0.784 | 0.742 | 94.6 M |
| 3 | wav2vec2-base + CREMA-D/TESS (seed 7) | 0.771 | 0.767 | 0.717 | 94.6 M |
| 4 | WavLM-large + ½ CREMA-D | 0.767 | 0.757 | 0.817 | 315.7 M |
| 5 | wav2vec2-base | 0.746 | 0.739 | 0.783 | 94.6 M |
| 6 | Ensemble (weights tuned on val) | 0.742 | 0.727 | 0.850 | – |
| 7 | WavLM-large + CREMA-D/TESS | 0.738 | 0.719 | 0.842 | 315.7 M |
| 8 | WavLM-large | 0.725 | 0.707 | 0.758 | 315.7 M |
| 9 | CNN + BiLSTM + Attention | 0.596 | 0.572 | 0.633 | 1.3 M |
| 10 | MLP (handcrafted) | 0.500 | 0.489 | 0.525 | 0.28 M |
| 11 | CNN (log-mel) | 0.454 | 0.445 | 0.608 | 1.2 M |

Per emotion (best model, test recall): surprised 100 %, neutral / calm / disgust 94 %, angry 88 %;
**happy, sad and fearful 59 %** (confused with surprised, calm and sad respectively).
Details and what was tried: `PROJECT_SUMMARY.md`.

Chance level = 12.5 % (8 classes). Confusion matrices: `results/<model>_confusion.png`;
learning curves: `results/<model>_curves.png`; chart: `results/comparison.png`.

## 7. Face fusion (optional)

* `--face-backend deepface`: pretrained DeepFace emotion model (`enforce_detection=False`; if no face
  is detected the face result is ignored).
* `--face-backend custom`: our FER-2013 CNN (`train_face_cnn.py`) + OpenCV Haar cascade face detector.
* `--face-backend auto`: DeepFace if installed, otherwise the custom model.

Face emotions map to audio classes: angry→angry, disgust→disgust, fear→fearful, happy→happy, sad→sad,
surprise→surprised, neutral→50 % neutral + 50 % calm. Fusion:
`fused = w_audio·audio + (1−w_audio)·face`, default `w_audio = 0.7`, and `w_audio < 0.5` is rejected.
In the live demo ~6 evenly spaced webcam frames captured *during* the recording are averaged.

## 8. Running on Google Colab (GPU)

1. Runtime → Change runtime type → **GPU**.
2. Upload this project folder as a zip (or push it to GitHub and `git clone` it), then in a cell:
   ```python
   !unzip -q ml_project.zip -d /content && %cd /content/ml_project
   !pip install -q -r requirements.txt -r requirements-optional.txt
   ```
3. Get RAVDESS with the Kaggle API (upload your `kaggle.json` from Kaggle → Settings → API):
   ```python
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p data
   !unzip -q data/ravdess-emotional-speech-audio.zip -d data/RAVDESS
   ```
4. Train and compare:
   ```python
   !python train.py --model all
   !python train_wav2vec2.py --epochs 20 --batch 8
   !python compare.py
   ```
5. Download `saved_models/` and `results/` (e.g. `!zip -r out.zip saved_models results`) to use the
   live demo on your laptop (the microphone/webcam demo must run locally, not on Colab).

## 9. Viva cheat-sheet

* **Why log-mel?** Mel scale matches human pitch perception; log compresses loudness like our ears.
* **Why attention?** Emotion is not spread evenly — a laugh or a raised voice lasts a few frames; attention
  learns per-frame weights instead of plain averaging.
* **Why class weights?** RAVDESS has no "strong" intensity for neutral, so neutral has 96 clips vs 192 for the others.
* **Why macro-F1 as well as accuracy?** It weights every emotion equally, so a model cannot look good by
  ignoring a small class.
* **Why late fusion?** Audio and face models are trained on different datasets; combining output
  probabilities needs no paired audio-visual data, and the weight keeps audio dominant.
