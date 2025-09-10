import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

epochs = data.get("epochs", [])
losses = data.get("losses", {})
metrics = data.get("metrics", {})
preds = data.get("predictions", [])
gts = data.get("ground_truth", [])


# Helper: compute accuracy from preds & gts lists-of-lists
def compute_accuracy(predictions, truths):
    acc = []
    for p, t in zip(predictions, truths):
        correct = np.equal(p, t).mean() if len(t) else np.nan
        acc.append(correct)
    return acc


try:  # 1. Loss curves
    if epochs and losses.get("train") and losses.get("val"):
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train")
        plt.plot(epochs, losses["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:  # 2. CRWA metric
    val_crwa = metrics.get("val_crwa", [])
    if epochs and val_crwa:
        plt.figure()
        plt.plot(epochs, val_crwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CRWA")
        plt.title("SPR_BENCH: Validation CRWA over Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_CRWA_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating CRWA plot: {e}")
    plt.close()

try:  # 3. Accuracy curve
    acc = compute_accuracy(preds, gts)
    if epochs and any(~np.isnan(acc_i) for acc_i in acc):
        plt.figure()
        plt.plot(epochs, acc, marker="s", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Validation Accuracy over Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating Accuracy plot: {e}")
    plt.close()
