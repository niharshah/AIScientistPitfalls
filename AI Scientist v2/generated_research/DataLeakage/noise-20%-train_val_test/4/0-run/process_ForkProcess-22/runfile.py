import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("no_transformer_context", {}).get("spr_bench", {})

epochs = ed.get("epochs", [])
tr_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
tr_f1 = ed.get("metrics", {}).get("train_f1", [])
val_f1 = ed.get("metrics", {}).get("val_f1", [])
test_f1 = ed.get("metrics", {}).get("test_f1", None)
preds = ed.get("predictions", [])
gts = ed.get("ground_truth", [])

# ------------------------------------------------------------------
# 1) Train / Val loss curve
# ------------------------------------------------------------------
try:
    if epochs and tr_loss and val_loss:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR-BENCH: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_train_val_loss_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Train / Val F1 curve
# ------------------------------------------------------------------
try:
    if epochs and tr_f1 and val_f1:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR-BENCH: Train vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_train_val_f1_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Final Test vs Best-Val F1 bar
# ------------------------------------------------------------------
try:
    if test_f1 is not None and val_f1:
        plt.figure()
        bars = ["Best Val F1", "Test F1"]
        vals = [max(val_f1), test_f1]
        plt.bar(bars, vals, color=["steelblue", "orange"])
        plt.ylim(0, 1)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.title("SPR-BENCH: Validation vs Test Macro-F1")
        fname = os.path.join(working_dir, "spr_bench_val_vs_test_f1_bar.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4) Confusion matrix
# ------------------------------------------------------------------
try:
    if preds and gts:
        cm = confusion_matrix(gts, preds)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        num_labels = cm.shape[0]
        ticks = np.arange(num_labels)
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR-BENCH: Confusion Matrix (Test Set)")
        for i in range(num_labels):
            for j in range(num_labels):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
