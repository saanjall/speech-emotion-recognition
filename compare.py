"""
compare.py - read results/*.json and compare all audio models.

    python compare.py
Prints a table sorted by test accuracy and saves:
    results/comparison.png   grouped bar chart (accuracy + macro-F1)
    results/comparison.md    the same table in Markdown (paste into the README / report)
"""
import os
import glob
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

PRETTY = {"mlp": "MLP (handcrafted)", "cnn": "CNN (log-mel)",
          "cnn_bilstm": "CNN+BiLSTM+Attn", "wav2vec2": "wav2vec2-base",
          "wavlm_large": "WavLM-large", "wav2vec2_extra": "wav2vec2-base + CREMA-D/TESS",
          "wav2vec2_tess": "wav2vec2-base + TESS", "wavlm_large_extra": "WavLM-large + 1/2 CREMA-D",
          "wavlm_large_full": "WavLM-large + CREMA-D/TESS", "wav2vec2_extra_s7": "wav2vec2-base + CREMA-D/TESS (seed 7)",
"ensemble": "Ensemble (weighted avg)", "ensemble_avg": "Ensemble (equal avg)"}


def load_results(results_dir=config.RESULTS_DIR):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if r.get("model") in config.AUDIO_MODELS + ["ensemble", "ensemble_avg"]:   # skip face_cnn and other files
            rows.append(r)
    return sorted(rows, key=lambda r: r["test_accuracy"], reverse=True)


def make_table(rows):
    header = "| Rank | Model | Test acc | Macro-F1 | Best val acc | Epochs | Params |\n" \
             "|---|---|---|---|---|---|---|"
    lines = [header]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {PRETTY.get(r['model'], r['model'])} | {r['test_accuracy']:.4f} | "
                     f"{r['macro_f1']:.4f} | {r.get('best_val_accuracy', float('nan')):.4f} | "
                     f"{r.get('epochs_trained', '-')} | {r.get('params', 0):,} |")
    return "\n".join(lines)


def plot(rows, path):
    names = [PRETTY.get(r["model"], r["model"]) for r in rows]
    acc = [r["test_accuracy"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - 0.2, acc, 0.4, label="Test accuracy")
    b2 = ax.bar(x + 0.2, f1, 0.4, label="Macro-F1")
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.axhline(1 / config.NUM_CLASSES, color="grey", ls="--", lw=1, label="chance (12.5%)")
    ax.set_xticks(x, names, rotation=25, ha="right")      # long names would overlap horizontally
    ax.set_ylim(0, 1)
    ax.set_title("RAVDESS speaker-independent test set (actors 21-24)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=config.RESULTS_DIR)
    args = parser.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        print(f"No model results in {args.results_dir}. Train first: python train.py --model all")
        return
    table = make_table(rows)
    print(table)
    print(f"\nBest model: {PRETTY.get(rows[0]['model'], rows[0]['model'])} "
          f"(test accuracy {rows[0]['test_accuracy']:.4f})")
    with open(os.path.join(args.results_dir, "comparison.md"), "w") as f:
        f.write(table + "\n")
    plot(rows, os.path.join(args.results_dir, "comparison.png"))
    print(f"Saved {os.path.join(args.results_dir, 'comparison.png')} and comparison.md")


if __name__ == "__main__":
    main()
