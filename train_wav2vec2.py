"""
train_wav2vec2.py - OPTIONAL transfer-learning model (PyTorch + HuggingFace).

wav2vec2-base was pre-trained (self-supervised) on 960 h of LibriSpeech, so it already
"understands" speech. We fine-tune it for 8-class emotion classification on the SAME
speaker-independent actor split as the Keras models, so the comparison is fair.

    * 16 kHz input (wav2vec2 requirement), same prepare_waveform() (trim/normalise/3 s)
    * CNN feature encoder frozen (saves memory, avoids over-fitting a small dataset)
    * light on-the-fly augmentation for TRAIN only: Gaussian noise + time shift
    * custom loop, mixed precision (AMP) when CUDA is available
    * best epoch selected on VALIDATION accuracy, test set evaluated once at the end
Designed for Google Colab (GPU). Run:  python train_wav2vec2.py --epochs 20 --batch 8
Requires: pip install -r requirements-optional.txt
"""
import os
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# Auto* classes load the right architecture for any checkpoint (wav2vec2, WavLM, HuBERT ...)
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm

import config
import data
from metrics_utils import save_results


def load_wave(path):
    return data.prepare_waveform(librosa.load(path, sr=config.W2V_SR, mono=True)[0], sr=config.W2V_SR)


class RavdessWaveDataset(Dataset):
    """Holds prepared 16 kHz waveforms in memory; augments on the fly if `train=True`."""

    def __init__(self, items, train=False, seed=config.SEED):
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.labels = np.array([lab for _, lab, _ in items], dtype=np.int64)
        print(f"  loading {len(items)} clips ({'train' if train else 'eval'}) ...")
        self.waves = Parallel(n_jobs=config.N_JOBS)(delayed(load_wave)(p) for p, _, _ in items)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        y = self.waves[idx].copy()
        if self.train:
            if self.rng.random() < 0.5:
                y = data.add_noise(y, self.rng)
            if self.rng.random() < 0.5:
                y = data.time_shift(y, self.rng)
        return y.astype(np.float32), int(self.labels[idx])


def make_collate(feature_extractor):
    def collate(batch):
        waves, labels = zip(*batch)
        inputs = feature_extractor(list(waves), sampling_rate=config.W2V_SR, return_tensors="pt", padding=True)
        return inputs["input_values"], torch.tensor(labels, dtype=torch.long)
    return collate


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x).logits
        total_loss += loss_fn(logits.float(), y).item() * len(y)
        preds.append(logits.argmax(-1).cpu().numpy())
        trues.append(y.cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return total_loss / len(trues), float((preds == trues).mean()), preds, trues


def main():
    parser = argparse.ArgumentParser(description="Fine-tune wav2vec2-base on RAVDESS (actor split).")
    parser.add_argument("--epochs", type=int, default=config.W2V_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.W2V_BATCH)
    parser.add_argument("--lr", type=float, default=config.W2V_LR)
    parser.add_argument("--patience", type=int, default=config.W2V_PATIENCE)
    # these override config.py for one run, so several runs can be queued without editing it
    parser.add_argument("--model", default=config.W2V_MODEL_NAME)
    parser.add_argument("--run", default=config.W2V_RUN_NAME, help="saved_models/<run>/ and results/<run>.json")
    parser.add_argument("--corpora", nargs="*", default=config.W2V_EXTRA_CORPORA)
    parser.add_argument("--fraction", type=float, default=config.W2V_EXTRA_FRACTION)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    config.W2V_MODEL_NAME, config.W2V_RUN_NAME, config.SEED = args.model, args.run, args.seed
    config.W2V_EXTRA_CORPORA, config.W2V_EXTRA_FRACTION = args.corpora, args.fraction
    OUT_DIR = os.path.join(config.MODELS_DIR, config.W2V_RUN_NAME)

    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    splits = data.split_by_actor(data.list_ravdess_files())
    train_items = splits["train"]
    if config.W2V_EXTRA_CORPORA:                 # extra corpora go into TRAIN only; val/test stay RAVDESS
        extra = data.list_extra_files(config.W2V_EXTRA_CORPORA)
        if config.W2V_EXTRA_FRACTION < 1:          # random subset for speed; keeps (almost) every speaker
            rng = np.random.default_rng(config.SEED)
            keep = rng.choice(len(extra), int(len(extra) * config.W2V_EXTRA_FRACTION), replace=False)
            extra = [extra[i] for i in sorted(keep)]
        print(f"extra training clips: {len(extra)} ({' + '.join(config.W2V_EXTRA_CORPORA)}) + {len(train_items)} RAVDESS")
        train_items = train_items + extra
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.W2V_MODEL_NAME)
    collate = make_collate(feature_extractor)
    train_ds = RavdessWaveDataset(train_items, train=True, seed=config.SEED)
    val_ds = RavdessWaveDataset(splits["val"])
    test_ds = RavdessWaveDataset(splits["test"])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=args.batch, collate_fn=collate)

    model = AutoModelForAudioClassification.from_pretrained(
        config.W2V_MODEL_NAME, num_labels=config.NUM_CLASSES,
        label2id={n: i for i, n in enumerate(config.EMOTIONS)},
        id2label={i: n for i, n in enumerate(config.EMOTIONS)})
    model.freeze_feature_encoder()            # keep the convolutional waveform encoder fixed
    model.to(device)

    weights = compute_class_weight("balanced", classes=np.arange(config.NUM_CLASSES), y=train_ds.labels)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_dl)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}
    best_val, bad_epochs, start = -1.0, 0, time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch:02d}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x).logits
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            seen += len(y)

        val_loss, val_acc, _, _ = evaluate(model, val_dl, device, loss_fn)
        history["loss"].append(run_loss / seen)
        history["accuracy"].append(correct / seen)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"epoch {epoch:02d} | train loss {run_loss / seen:.4f} acc {correct / seen:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val:                 # keep the best model on VALIDATION accuracy
            best_val, bad_epochs = val_acc, 0
            model.save_pretrained(OUT_DIR)
            feature_extractor.save_pretrained(OUT_DIR)
            print(f"  -> new best, saved to {OUT_DIR}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("  early stopping")
                break

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"model": config.W2V_RUN_NAME, "base": config.W2V_MODEL_NAME, "framework": "torch", "sr": config.W2V_SR,
                   "duration": config.DURATION, "class_names": config.EMOTIONS}, f, indent=2)

    # Reload the best checkpoint and evaluate ONCE on the test actors.
    model = AutoModelForAudioClassification.from_pretrained(OUT_DIR).to(device)
    _, _, preds, trues = evaluate(model, test_dl, device, loss_fn)
    save_results(config.W2V_RUN_NAME, trues, preds, history=history,
                 extra={"params": int(sum(p.numel() for p in model.parameters())),
                        "train_time_sec": round(time.time() - start, 1),
                        "train_samples": len(train_ds)})


if __name__ == "__main__":
    main()
