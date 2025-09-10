import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data & compute final metrics -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}


# Helper to compute metrics
def complexity_weighted_acc(preds, gts, wts):
    correct = (preds == gts).astype(float)
    return (correct * wts).sum() / wts.sum()


try:
    preds = np.array(exp.get("predictions", []))
    gts = np.array(exp.get("ground_truth", []))
    wts = np.array(exp.get("weights", []))
    test_acc = (preds == gts).mean() if len(preds) else np.nan
    test_cwa = complexity_weighted_acc(preds, gts, wts) if len(preds) else np.nan
    print(f"Test Accuracy: {test_acc:.4f}, Test CWA: {test_cwa:.4f}")
except Exception as e:
    print(f"Error computing metrics: {e}")

# --------------------------- plotting -------------------------------
# 1) Loss curves
try:
    plt.figure()
    epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
    plt.plot(epochs, exp["losses"]["train"], label="Train Loss")
    plt.plot(epochs, exp["losses"]["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Accuracy & CWA curves
try:
    plt.figure()
    epochs = np.arange(1, len(exp["metrics"]["accuracy"]) + 1)
    plt.plot(epochs, exp["metrics"]["accuracy"], label="Val Accuracy")
    plt.plot(epochs, exp["metrics"]["cwa"], label="Val CWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation Accuracy & CWA")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_cwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy/CWA curves: {e}")
    plt.close()

# 3) Confusion matrix
try:
    cm = np.zeros((2, 2), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH Confusion Matrix (Test)")
    plt.colorbar()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 4) Complexity vs correctness scatter
try:
    correctness = (preds == gts).astype(int)
    plt.figure()
    plt.scatter(wts, correctness, alpha=0.4, s=10)
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.xlabel("Sequence Complexity (weight)")
    plt.ylabel("Prediction Outcome")
    plt.title("SPR_BENCH: Correctness vs Complexity (Test)")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_complexity_scatter.png"))
    plt.close()
except Exception as e:
    print(f"Error creating complexity scatter: {e}")
    plt.close()
