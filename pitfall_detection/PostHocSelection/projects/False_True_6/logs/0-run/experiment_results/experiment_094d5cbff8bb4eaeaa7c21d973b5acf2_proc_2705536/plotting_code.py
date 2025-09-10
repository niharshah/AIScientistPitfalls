import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths & load ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# ----- pick the single exp entry ----------
try:
    rec = experiment_data["NoSymFeatTransformer"]["SPR_BENCH"]
except KeyError:
    print("Expected keys not found in experiment_data.")
    exit()

loss_train = rec["losses"]["train"]
loss_val = rec["losses"]["val"]
swa_val = rec["SWA"]["val"]
test_metrics = rec["metrics"]["test"]
preds = np.array(rec.get("predictions", []))
trues = np.array(rec.get("ground_truth", []))

# ---------------- PLOT 1: loss curves ---------------
try:
    plt.figure(figsize=(6, 4))
    epochs = np.arange(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------- PLOT 2: SWA curve -----------------
try:
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, swa_val, marker="o", label="Val SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH: Validation SWA per Epoch")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_swa_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------------- PLOT 3: confusion matrix ----------
try:
    if preds.size and trues.size and preds.shape == trues.shape:
        n_labels = len(set(trues.tolist() + preds.tolist()))
        cm = np.zeros((n_labels, n_labels), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.xticks(range(n_labels))
        plt.yticks(range(n_labels))
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- print final metrics ---------------
print(
    f"Test metrics -> Loss: {test_metrics.get('loss', 'NA'):.4f}, "
    f"SWA: {test_metrics.get('SWA', 'NA'):.3f}"
)
