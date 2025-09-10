import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("char_bigram_transformer", {})
epochs = np.asarray(ed.get("epochs", []))
train_losses = np.asarray(ed.get("losses", {}).get("train", []))
val_losses = np.asarray(ed.get("losses", {}).get("val", []))
train_f1 = np.asarray(ed.get("metrics", {}).get("train_f1", []))
val_f1 = np.asarray(ed.get("metrics", {}).get("val_f1", []))
test_f1 = ed.get("metrics", {}).get("test_f1", None)
preds = np.asarray(ed.get("predictions", []))
gts = np.asarray(ed.get("ground_truth", []))

# 1) Loss curve
try:
    if epochs.size and train_losses.size and val_losses.size:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.title("Train vs Validation Loss - SPR_BENCH (char_bigram_transformer)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_char_bigram_loss_curve.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) F1 curve
try:
    if epochs.size and train_f1.size and val_f1.size:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.title("Train vs Validation Macro-F1 - SPR_BENCH (char_bigram_transformer)")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_char_bigram_f1_curve.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 3) Histogram of test label distribution
try:
    if preds.size and gts.size:
        plt.figure(figsize=(8, 4))
        labels = np.unique(np.concatenate([gts, preds]))
        plt.subplot(1, 2, 1)
        gt_counts = [np.sum(gts == l) for l in labels]
        plt.bar(labels, gt_counts)
        plt.title("Left: Ground Truth Label Counts - SPR_BENCH")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.subplot(1, 2, 2)
        pred_counts = [np.sum(preds == l) for l in labels]
        plt.bar(labels, pred_counts, color="orange")
        plt.title("Right: Predicted Label Counts - SPR_BENCH")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_char_bigram_label_hist.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating label histogram: {e}")
    plt.close()

# 4) Confusion matrix heatmap
try:
    if preds.size and gts.size:
        labels = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((labels.size, labels.size), dtype=int)
        for p, t in zip(preds, gts):
            cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(labels.size), labels, rotation=90)
        plt.yticks(range(labels.size), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix - SPR_BENCH (char_bigram_transformer)")
        fname = os.path.join(working_dir, "SPR_BENCH_char_bigram_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# Print final evaluation metric
if test_f1 is not None:
    print(f"Final Test Macro-F1: {test_f1:.4f}")
