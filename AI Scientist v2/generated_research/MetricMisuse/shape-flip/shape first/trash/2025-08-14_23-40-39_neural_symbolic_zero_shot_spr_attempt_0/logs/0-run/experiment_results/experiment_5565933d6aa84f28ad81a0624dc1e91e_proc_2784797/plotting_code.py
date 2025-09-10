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
    exit()

ed = experiment_data["TokenOrderShuffling"]["SPR_BENCH"]
metrics = ed["metrics"]

# Plot 1: Train vs Validation Loss
try:
    epochs = np.arange(1, len(metrics["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.title("SPR_BENCH – Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# Plot 2: Validation Shape-Weighted Accuracy
try:
    plt.figure()
    plt.plot(epochs, metrics["val_swa"], marker="o", color="green")
    plt.title("SPR_BENCH – Validation Shape-Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    fname = os.path.join(working_dir, "SPR_BENCH_val_swa_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# Plot 3: Dev vs Test Accuracy Bar Chart
try:
    dev_pred = np.array(ed["predictions"]["dev"])
    dev_gt = np.array(ed["ground_truth"]["dev"])
    test_pred = np.array(ed["predictions"]["test"])
    test_gt = np.array(ed["ground_truth"]["test"])

    dev_acc = float((dev_pred == dev_gt).mean()) if dev_pred.size else 0.0
    test_acc = float((test_pred == test_gt).mean()) if test_pred.size else 0.0

    plt.figure()
    plt.bar(["Dev", "Test"], [dev_acc, test_acc], color=["steelblue", "orange"])
    plt.ylim(0, 1)
    plt.title("SPR_BENCH – Accuracy on Dev vs Test")
    for i, v in enumerate([dev_acc, test_acc]):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fname = os.path.join(working_dir, "SPR_BENCH_dev_test_accuracy.png")
    plt.savefig(fname)
    plt.close()

    print(f"Final Dev Accuracy : {dev_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
except Exception as e:
    print(f"Error creating accuracy bar chart: {e}")
    plt.close()
