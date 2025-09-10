import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})
metrics = spr.get("metrics", {})
losses = spr.get("losses", {})
preds = spr.get("predictions", [])
golds = spr.get("ground_truth", [])

epochs = range(1, len(metrics.get("train_acc", [])) + 1)

# ---- 1. Train / Val Accuracy ------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train_acc", []), label="Train Acc")
    plt.plot(epochs, metrics.get("val_acc", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs. Validation Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---- 2. Train / Val Loss ----------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train Loss")
    plt.plot(epochs, losses.get("val", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---- 3. URA_val across epochs ----------------------------------------------
try:
    ura_vals = metrics.get("URA_val", [])
    if ura_vals:
        plt.figure()
        plt.plot(epochs, ura_vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("URA")
        plt.title("SPR_BENCH: Unseen Reconstruction Accuracy (URA_val)")
        fname = os.path.join(working_dir, "SPR_BENCH_URA_val_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating URA plot: {e}")
    plt.close()

# ---- 4. Confusion Matrix ----------------------------------------------------
try:
    if preds and golds:
        labels = sorted(list(set(golds) | set(preds)))
        label2idx = {l: i for i, l in enumerate(labels)}
        mat = np.zeros((len(labels), len(labels)), dtype=int)
        for g, p in zip(golds, preds):
            mat[label2idx[g], label2idx[p]] += 1
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---- Print final evaluation metrics ----------------------------------------
val_loss_last = (
    metrics.get("val_loss", [np.nan])[-1] if metrics.get("val_loss") else np.nan
)
ura_test = metrics.get("URA_test", [np.nan])[-1] if metrics.get("URA_test") else np.nan
val_acc_last = (
    metrics.get("val_acc", [np.nan])[-1] if metrics.get("val_acc") else np.nan
)
print(
    f"Final Val Acc: {val_acc_last:.4f} | Final Val Loss: {val_loss_last:.4f} | URA_test: {ura_test:.4f}"
)
