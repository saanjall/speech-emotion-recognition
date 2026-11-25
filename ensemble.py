"""
ensemble.py - weighted average of the trained models' probabilities (late fusion of audio models).

    fused = sum_m w_m * probs_m       weights w_m >= 0, sum to 1 (grid step 0.1)

The weights are chosen on the VALIDATION actors (19-20) only; the TEST actors (21-24)
are scored once with the chosen weights, so the reported accuracy stays honest.
Different models make different mistakes, so averaging them usually beats the best single one.

    python ensemble.py
    python ensemble.py --equal wav2vec2_extra wavlm_large_extra   # plain average, nothing tuned on validation
"""
import os
import argparse
import itertools

os.environ.setdefault("KERAS_BACKEND", "torch")      # same backend as train.py

import numpy as np

import config
import data
from audio_predictor import AudioEmotionPredictor
from metrics_utils import save_results

GRID_STEPS = 10                                      # weights in multiples of 0.1


def model_probs(name, items):
    """Probabilities (N, 8) for each (path, label, actor) item; cached in cache/probs_<name>_<split>.npz."""
    split = "val" if items[0][2] in config.VAL_ACTORS else "test"
    path = os.path.join(config.CACHE_DIR, f"probs_{name}_{split}.npz")
    model_dir = os.path.join(config.MODELS_DIR, name)
    model_time = max(os.path.getmtime(os.path.join(model_dir, f)) for f in os.listdir(model_dir))
    paths = np.array([p for p, _, _ in items])
    if os.path.exists(path):
        c = np.load(path)
        if c["model_time"] == model_time and np.array_equal(c["paths"], paths):   # model not retrained since
            return c["probs"]
    print(f"  predicting {len(items)} {split} clips with {name} ...")
    pred = AudioEmotionPredictor(name)
    probs = np.stack([pred.predict_file(p) for p in paths])
    np.savez(path, probs=probs, paths=paths, model_time=model_time)
    return probs


def weight_grid(n):
    """All weight vectors of length n with non-negative multiples of 1/GRID_STEPS summing to 1."""
    for combo in itertools.product(range(GRID_STEPS + 1), repeat=n):
        if sum(combo) == GRID_STEPS:
            yield np.array(combo) / GRID_STEPS


def main():
    parser = argparse.ArgumentParser(description="Average trained audio models' probabilities.")
    parser.add_argument("--equal", nargs="+", metavar="MODEL",
                        help="average just these models with equal weights (saved as 'ensemble_avg')")
    args = parser.parse_args()
    names = args.equal or [m for m in config.AUDIO_MODELS if os.path.isdir(os.path.join(config.MODELS_DIR, m))]
    print("models:", names)
    splits = data.split_by_actor(data.list_ravdess_files())
    y_val = np.array([lab for _, lab, _ in splits["val"]])
    y_test = np.array([lab for _, lab, _ in splits["test"]])
    P_val = np.stack([model_probs(m, splits["val"]) for m in names])      # (M, N_val, 8)
    P_test = np.stack([model_probs(m, splits["test"]) for m in names])

    print("\nsingle models     val acc   test acc")
    for m, pv, pt in zip(names, P_val, P_test):
        print(f"  {m:14s}  {(pv.argmax(1) == y_val).mean():.4f}    {(pt.argmax(1) == y_test).mean():.4f}")

    # Pick weights on VALIDATION: best accuracy, ties broken by lower log-loss.
    def val_score(w):
        fused = np.tensordot(w, P_val, axes=1)
        nll = -np.log(fused[np.arange(len(y_val)), y_val] + 1e-9).mean()
        return (fused.argmax(1) == y_val).mean(), -nll

    if args.equal:
        best_w = np.full(len(names), 1 / len(names))
    else:
        best_w = max(weight_grid(len(names)), key=val_score)
    val_acc = val_score(best_w)[0]
    y_pred = np.tensordot(best_w, P_test, axes=1).argmax(1)              # test used ONCE

    print("\nweights: " + ", ".join(f"{m}={w:.2f}" for m, w in zip(names, best_w) if w > 0))
    print(f"ensemble  val acc {val_acc:.4f}  ->  TEST acc {(y_pred == y_test).mean():.4f}")
    save_results("ensemble_avg" if args.equal else "ensemble", y_test, y_pred,
                 extra={"best_val_accuracy": float(val_acc),
                        "weights": {m: float(w) for m, w in zip(names, best_w)}})


if __name__ == "__main__":
    main()
