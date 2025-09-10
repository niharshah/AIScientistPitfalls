import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_runs = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})

# ----------------- gather per-run data -----------------
loss_curves, f1_curves = {}, {}
best_run_key, best_val_f1 = None, -1
for key, rec in spr_runs.items():
    epochs = np.array(rec.get("epochs", []))
    tr_loss = np.array(rec.get("losses", {}).get("train", []))
    val_loss = np.array(rec.get("losses", {}).get("val", []))
    tr_f1 = np.array(rec.get("metrics", {}).get("train_f1", []))
    val_f1 = np.array(rec.get("metrics", {}).get("val_f1", []))
    if len(epochs):
        loss_curves[key] = (epochs, tr_loss, val_loss)
        f1_curves[key] = (epochs, tr_f1, val_f1)
        if val_f1.max() > best_val_f1:
            best_val_f1 = val_f1.max()
            best_run_key = key

print(f"Best run: {best_run_key} with Dev F1={best_val_f1:.4f}")

# ------------- plot 1: loss curves -------------
try:
    plt.figure()
    for key, (ep, tr, val) in loss_curves.items():
        plt.plot(ep, tr, label=f"{key} train")
        plt.plot(ep, val, "--", label=f"{key} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Loss Curves (all learning rates)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------- plot 2: F1 curves -------------
try:
    plt.figure()
    for key, (ep, tr, val) in f1_curves.items():
        plt.plot(ep, tr, label=f"{key} train")
        plt.plot(ep, val, "--", label=f"{key} val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH – F1 Curves (all learning rates)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# ------------- plot 3: confusion matrix for best run -------------
try:
    from sklearn.metrics import confusion_matrix

    if best_run_key:
        preds = np.array(spr_runs[best_run_key]["predictions"])
        gts = np.array(spr_runs[best_run_key]["ground_truth"])
        cm = confusion_matrix(gts, preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title(f"SPR_BENCH – Confusion Matrix (best LR: {best_run_key})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_{best_run_key}.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
