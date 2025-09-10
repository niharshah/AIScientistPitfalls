import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
base_data = experiment_data.get(dataset, {}).get("baseline", {})
symb_data = experiment_data.get(dataset, {}).get("symbolic", {})


def _collect(d, key):
    return [x[key] for x in d] if d else []


epochs_b = _collect(base_data.get("losses", {}).get("train", []), "epoch")
epochs_s = _collect(symb_data.get("losses", {}).get("train", []), "epoch")

loss_tr_b = _collect(base_data.get("losses", {}).get("train", []), "loss")
loss_val_b = _collect(base_data.get("losses", {}).get("val", []), "loss")
loss_tr_s = _collect(symb_data.get("losses", {}).get("train", []), "loss")
loss_val_s = _collect(symb_data.get("losses", {}).get("val", []), "loss")

f1_tr_b = _collect(base_data.get("metrics", {}).get("train", []), "macro_f1")
f1_val_b = _collect(base_data.get("metrics", {}).get("val", []), "macro_f1")
f1_tr_s = _collect(symb_data.get("metrics", {}).get("train", []), "macro_f1")
f1_val_s = _collect(symb_data.get("metrics", {}).get("val", []), "macro_f1")

# --------------------------------- 1) Loss curves
try:
    plt.figure()
    if epochs_b:
        plt.plot(epochs_b, loss_tr_b, "--", label="Baseline-Train")
        plt.plot(epochs_b, loss_val_b, label="Baseline-Val")
    if epochs_s:
        plt.plot(epochs_s, loss_tr_s, "--", label="Symbolic-Train")
        plt.plot(epochs_s, loss_val_s, label="Symbolic-Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss\nDataset: SPR_BENCH")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "SPR_BENCH_baseline_vs_symbolic_loss_curves.png")
    )
    plt.close()
except Exception as e:
    print(f"Error plotting loss curves: {e}")
    plt.close()

# --------------------------------- 2) Macro-F1 curves
try:
    plt.figure()
    if epochs_b:
        plt.plot(epochs_b, f1_tr_b, "--", label="Baseline-Train")
        plt.plot(epochs_b, f1_val_b, label="Baseline-Val")
    if epochs_s:
        plt.plot(epochs_s, f1_tr_s, "--", label="Symbolic-Train")
        plt.plot(epochs_s, f1_val_s, label="Symbolic-Val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training vs Validation Macro-F1\nDataset: SPR_BENCH")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "SPR_BENCH_baseline_vs_symbolic_f1_curves.png")
    )
    plt.close()
except Exception as e:
    print(f"Error plotting F1 curves: {e}")
    plt.close()

# --------------------------------- 3) Final validation Macro-F1 bar chart
try:
    plt.figure()
    labels, vals = [], []
    if f1_val_b:
        labels.append("Baseline")
        vals.append(f1_val_b[-1])
    if f1_val_s:
        labels.append("Symbolic")
        vals.append(f1_val_s[-1])
    xs = np.arange(len(labels))
    plt.bar(xs, vals, tick_label=labels)
    plt.ylabel("Final Val Macro-F1")
    plt.title("Final Validation Macro-F1 Comparison\nDataset: SPR_BENCH")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_val_f1_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error plotting bar chart: {e}")
    plt.close()

# --------------------------------- 4) Confusion matrix for Symbolic model
try:
    preds = symb_data.get("predictions", [])
    gts = symb_data.get("ground_truth", [])
    if preds and gts:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix – Symbolic Model\nDataset: SPR_BENCH")
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_symbolic_confusion_matrix.png")
        )
        plt.close()
except Exception as e:
    print(f"Error plotting confusion matrix: {e}")
    plt.close()

# --------------------------------- Print final validation scores
if f1_val_b:
    print(f"Baseline final val Macro-F1: {f1_val_b[-1]:.4f}")
if f1_val_s:
    print(f"Symbolic final val Macro-F1: {f1_val_s[-1]:.4f}")
