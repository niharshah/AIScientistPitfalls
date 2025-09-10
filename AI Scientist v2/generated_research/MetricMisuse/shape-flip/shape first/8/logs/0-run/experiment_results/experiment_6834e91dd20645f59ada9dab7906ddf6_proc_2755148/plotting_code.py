import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})
epochs = spr.get("epochs", [])
train_loss = spr.get("losses", {}).get("train", [])
train_swa = spr.get("metrics", {}).get("train_swa", [])
val_swa = spr.get("metrics", {}).get("val_swa", [])
test_swa = spr.get("metrics", {}).get("test_swa", None)
preds = np.array(spr.get("predictions", []))
gts = np.array(spr.get("ground_truth", []))

# ---------- plot 1: training loss ----------
try:
    if epochs and train_loss:
        plt.figure()
        plt.plot(epochs, train_loss, color="tab:blue")
        plt.title("SPR_BENCH – Training Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fname = os.path.join(working_dir, "SPR_BENCH_training_loss.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# ---------- plot 2: SWA curves ----------
try:
    if epochs and train_swa and val_swa:
        plt.figure()
        plt.plot(epochs, train_swa, label="Train SWA", color="tab:green")
        plt.plot(epochs, val_swa, label="Validation SWA", color="tab:orange")
        plt.title("SPR_BENCH – Shape-Weighted Accuracy vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_SWA_train_vs_val.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------- plot 3: final test SWA bar ----------
try:
    if test_swa is not None:
        plt.figure()
        plt.bar([0], [test_swa], color="tab:purple")
        plt.xticks([0], ["SPR_BENCH"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH – Final Test SWA")
        plt.ylabel("SWA")
        fname = os.path.join(working_dir, "SPR_BENCH_test_SWA_bar.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating test SWA bar plot: {e}")
    plt.close()

# ---------- plot 4: confusion matrix ----------
try:
    if preds.size and gts.size and preds.shape == gts.shape:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title("SPR_BENCH – Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print evaluation metrics ----------
if test_swa is not None and preds.size:
    acc = (preds == gts).mean()
    print(f"SPR_BENCH – Test SWA: {test_swa:.3f},  Accuracy: {acc:.3f}")
