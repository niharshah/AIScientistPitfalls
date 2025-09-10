import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp = experiment_data.get("no_proj", {}).get("SPR_BENCH", {})

loss_tr = np.array(exp.get("losses", {}).get("train", []), dtype=float)
loss_val = np.array(exp.get("losses", {}).get("val", []), dtype=float)
ccwa_val = np.array(exp.get("metrics", {}).get("val_CCWA", []), dtype=float)
pred_epochs = exp.get("predictions", [])
gt_epochs = exp.get("ground_truth", [])

epochs = np.arange(1, len(loss_tr) + 1)

# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) CCWA curve
try:
    plt.figure()
    plt.plot(epochs, ccwa_val, marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("CCWA")
    plt.title("SPR_BENCH Validation CCWA over Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_CCWA_curve.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()

# 3) Confusion matrix for final epoch
try:
    if pred_epochs and gt_epochs:
        preds = np.array(pred_epochs[-1])
        gts = np.array(gt_epochs[-1])
        classes = np.union1d(preds, gts)
        cm = np.zeros((classes.size, classes.size), dtype=int)
        for p, g in zip(preds, gts):
            cm[g == classes, p == classes] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix (Last Epoch)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# 4) Class frequency bars (last epoch)
try:
    if pred_epochs and gt_epochs:
        preds = np.array(pred_epochs[-1])
        gts = np.array(gt_epochs[-1])
        classes = np.union1d(preds, gts)
        pred_counts = [(preds == c).sum() for c in classes]
        gt_counts = [(gts == c).sum() for c in classes]
        x = np.arange(len(classes))
        plt.figure()
        plt.bar(x - 0.2, gt_counts, width=0.4, label="Ground Truth")
        plt.bar(x + 0.2, pred_counts, width=0.4, label="Predictions")
        plt.xticks(x, classes)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(
            "SPR_BENCH Class Distribution (Last Epoch)\nLeft: Ground Truth, Right: Predictions"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_class_distribution.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()
