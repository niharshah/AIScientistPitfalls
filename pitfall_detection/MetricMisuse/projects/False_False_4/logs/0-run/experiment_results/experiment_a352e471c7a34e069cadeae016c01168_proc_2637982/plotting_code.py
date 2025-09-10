import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("spr_bench", {})
metrics = spr.get("metrics", {})
losses = spr.get("losses", {})
swa = spr.get("swa", {})
test_metrics = spr.get("test_metrics", {})
epochs = range(1, len(losses.get("train", [])) + 1)

# ---------- Plot 1: Loss ----------
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="train")
    plt.plot(epochs, losses.get("val", []), label="val", linestyle="--")
    plt.title("SPR_Bench Loss Curves\nTrain vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_Bench_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- Plot 2: Accuracy ----------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train", []), label="train")
    plt.plot(epochs, metrics.get("val", []), label="val", linestyle="--")
    plt.title("SPR_Bench Accuracy Curves\nTrain vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_Bench_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- Plot 3: Shape-Weighted Accuracy ----------
try:
    plt.figure()
    plt.plot(epochs, swa.get("train", []), label="train")
    plt.plot(epochs, swa.get("val", []), label="val", linestyle="--")
    plt.title("SPR_Bench SWA Curves\nTrain vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_Bench_swa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------- Plot 4: Test metrics bar ----------
try:
    labels = ["Accuracy", "SWA"]
    values = [test_metrics.get("acc", 0), test_metrics.get("swa", 0)]
    plt.figure()
    plt.bar(np.arange(len(values)), values, color=["skyblue", "lightgreen"])
    plt.xticks(np.arange(len(values)), labels)
    plt.title("SPR_Bench Final Test Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(working_dir, "SPR_Bench_test_metrics_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar plot: {e}")
    plt.close()

# ---------- Plot 5: Confusion matrix ----------
try:
    preds = spr.get("predictions", {}).get("test", [])
    gts = spr.get("ground_truth", {}).get("test", [])
    if preds and gts:
        num_cls = max(max(preds), max(gts)) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_Bench Confusion Matrix\nTest Set")
        plt.savefig(os.path.join(working_dir, "SPR_Bench_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print final metrics ----------
if test_metrics:
    print(
        f"Test Loss: {test_metrics.get('loss', 'NA'):.4f}, "
        f"Accuracy: {test_metrics.get('acc', 'NA'):.3f}, "
        f"SWA: {test_metrics.get('swa', 'NA'):.3f}"
    )
