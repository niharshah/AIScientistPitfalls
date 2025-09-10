import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp = experiment_data.get("no_cluster_raw_vocab", {}).get("SPR_BENCH", {})

metrics = exp.get("metrics", {})
train_loss = metrics.get("train_loss", [])
val_loss = metrics.get("val_loss", [])
val_CWA = metrics.get("val_CWA", [])
val_SWA = metrics.get("val_SWA", [])
val_CWA2 = metrics.get("val_CWA2", [])

preds = np.array(exp.get("predictions", []))
gts = np.array(exp.get("ground_truth", []))

epochs = np.arange(1, len(train_loss) + 1)

# ------------------ plot 1: loss curves -------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss (Glyph Sequence Classification)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------ plot 2: weighted accuracies -----------
try:
    plt.figure()
    plt.plot(epochs, val_CWA, label="Color Weighted Acc.")
    plt.plot(epochs, val_SWA, label="Shape Weighted Acc.")
    plt.plot(epochs, val_CWA2, label="Complexity Weighted Acc.")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Weighted Accuracies over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_weighted_accuracies.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------ plot 3: confusion matrix --------------
try:
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("SPR_BENCH: Confusion Matrix (Dev)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

print(f"Plots saved to {working_dir}")
