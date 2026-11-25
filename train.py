"""
train.py - train and evaluate the Keras models on the speaker-independent split.

Examples
    python train.py --model all
    python train.py --model cnn_bilstm --epochs 80 --batch 32
    python train.py --model mlp --rebuild          # re-extract cached features first

For each model this saves to saved_models/<model>/:
    model.keras    trained network (best epoch on VALIDATION accuracy)
    scaler.joblib  (MLP)  StandardScaler fitted on TRAIN features only
    norm.json      (CNNs) per-mel-bin mean/std computed on TRAIN spectrograms only
    meta.json      everything inference needs (feature type, sample rate, class names)
and writes test metrics/plots to results/ via metrics_utils.save_results().
"""
import os
import json
import time
import argparse

# PyTorch backend: TensorFlow has no GPU support on native Windows, PyTorch does (~5-7x faster here).
# Must be set before keras is imported. Override with: $env:KERAS_BACKEND="tensorflow"
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import keras                      # Keras 3 (installed with TensorFlow >= 2.16)

import config
import data
import models
from metrics_utils import save_results

KERAS_MODELS = ["mlp", "cnn", "cnn_bilstm"]


def prepare_inputs(name, dataset, out_dir):
    """Return (X_train, y_train), (X_val, y_val), (X_test, y_test) normalised with TRAIN statistics."""
    feature = "vec" if name == "mlp" else "mel"
    (Xtr, ytr), (Xva, yva), (Xte, yte) = [data.get_split(dataset, s, feature) for s in ("train", "val", "test")]

    if name == "mlp":
        scaler = StandardScaler().fit(Xtr)                 # fit on TRAIN only -> no leakage
        joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
        Xtr, Xva, Xte = (scaler.transform(X) for X in (Xtr, Xva, Xte))
    else:
        norm = data.SpecNormalizer().fit(Xtr)              # fit on TRAIN only -> no leakage
        norm.save(os.path.join(out_dir, "norm.json"))
        Xtr, Xva, Xte = (norm.transform(X)[..., None] for X in (Xtr, Xva, Xte))   # add channel axis
    return (Xtr.astype("float32"), ytr), (Xva.astype("float32"), yva), (Xte.astype("float32"), yte)


def train_one(name, dataset, epochs=config.EPOCHS, batch=config.BATCH_SIZE,
              models_dir=config.MODELS_DIR, results_dir=config.RESULTS_DIR, verbose=2):
    print(f"\n################  Training {name}  ################")
    keras.utils.set_random_seed(config.SEED)
    out_dir = os.path.join(models_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare_inputs(name, dataset, out_dir)
    print(f"train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

    # Balanced class weights: 'neutral' has half as many clips as the other emotions in RAVDESS.
    classes = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = models.build_model(name, Xtr.shape[1:])
    model.summary(print_fn=lambda s: print(s) if verbose else None)

    callbacks = [
        # Model selection uses the VALIDATION actors; the test actors are never looked at here.
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config.EARLY_STOP_PATIENCE,
                                      mode="max", restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config.LR_PATIENCE,
                                          min_lr=1e-6, verbose=1),
    ]
    start = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch,
                     class_weight=class_weight, callbacks=callbacks, verbose=verbose)
    train_time = time.time() - start

    model.save(os.path.join(out_dir, "model.keras"))
    meta = {
        "model": name,
        "framework": "keras",
        "feature": "vec" if name == "mlp" else "mel",
        "sr": config.SR,
        "duration": config.DURATION,
        "class_names": config.EMOTIONS,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Final, one-time evaluation on the held-out TEST actors.
    y_pred = model.predict(Xte, batch_size=batch, verbose=0).argmax(axis=1)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return save_results(name, yte, y_pred, history=history, results_dir=results_dir,
                        extra={"params": int(model.count_params()),
                               "train_time_sec": round(train_time, 1),
                               "train_samples": int(len(ytr))})


def main():
    parser = argparse.ArgumentParser(description="Train speech emotion models on RAVDESS.")
    parser.add_argument("--model", default="all", choices=KERAS_MODELS + ["all"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--rebuild", action="store_true", help="re-extract features instead of using the cache")
    args = parser.parse_args()

    dataset = data.build_dataset(rebuild=args.rebuild)
    names = KERAS_MODELS if args.model == "all" else [args.model]
    summary = {n: train_one(n, dataset, args.epochs, args.batch) for n in names}

    print("\nmodel        test_acc  macro_f1")
    for n, r in summary.items():
        print(f"{n:12s} {r['test_accuracy']:.4f}    {r['macro_f1']:.4f}")
    print("Run `python compare.py` to build the comparison table/chart.")


if __name__ == "__main__":
    main()
