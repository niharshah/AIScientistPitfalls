import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths & data ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

run = experiment_data.get("no_color_emb", {})
metrics = run.get("metrics", {})
swa = run.get("swa", {})
losses = run.get("losses", {})
test_metrics = run.get("test_metrics", {})
pred_val = np.array(run.get("predictions", {}).get("val", []))
gt_val = np.array(run.get("ground_truth", {}).get("val", []))

epochs = np.arange(1, len(metrics.get("train", [])) + 1)

# ---------- accuracy ----------
try:
    plt.figure()
    plt.plot(epochs, metrics["train"], label="Train")
    plt.plot(epochs, metrics["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR (no_color_emb) Accuracy per Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_no_color_emb_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- SWA ----------
try:
    plt.figure()
    plt.plot(epochs, swa["train"], label="Train")
    plt.plot(epochs, swa["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR (no_color_emb) Shape-Weighted Accuracy per Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_no_color_emb_swa.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------- loss ----------
try:
    plt.figure()
    plt.plot(epochs, losses["train"], label="Train")
    plt.plot(epochs, losses["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR (no_color_emb) Loss per Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_no_color_emb_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- confusion matrix ----------
try:
    if pred_val.size and gt_val.size:
        num_classes = int(max(gt_val.max(), pred_val.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gt_val, pred_val):
            cm[t, p] += 1

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SPR (no_color_emb) Confusion Matrix (Validation)")
        plt.savefig(os.path.join(working_dir, "SPR_no_color_emb_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print test metrics ----------
if test_metrics:
    print(
        f"Test results - Loss: {test_metrics.get('loss'):.4f}, "
        f"Accuracy: {test_metrics.get('acc'):.3f}, "
        f"SWA: {test_metrics.get('swa'):.3f}"
    )
