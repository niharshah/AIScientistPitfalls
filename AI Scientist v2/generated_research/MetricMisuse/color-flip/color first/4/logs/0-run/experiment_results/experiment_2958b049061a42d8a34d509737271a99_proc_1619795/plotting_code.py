import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load experiment data -------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    quit()

epochs = np.arange(1, len(data["losses"]["train"]) + 1)

# ------------- plot losses -------------
try:
    plt.figure()
    plt.plot(epochs, data["losses"]["train"], label="Train Loss")
    plt.plot(epochs, data["losses"]["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------- plot metrics -------------
try:
    metrics = data["metrics"]["val"]
    acc = [m["acc"] for m in metrics]
    cwa = [m["cwa"] for m in metrics]
    swa = [m["swa"] for m in metrics]
    pcwa = [m["pcwa"] for m in metrics]

    plt.figure()
    plt.plot(epochs, acc, label="Accuracy")
    plt.plot(epochs, cwa, label="Color-Weighted Acc")
    plt.plot(epochs, swa, label="Shape-Weighted Acc")
    plt.plot(epochs, pcwa, label="PC-Weighted Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation Metrics over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_metrics_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ------------- print final metrics -------------
try:
    last = metrics[-1]
    print("Final Validation Metrics:")
    for k, v in last.items():
        if k != "epoch":
            print(f"  {k}: {v:.4f}")
except Exception as e:
    print(f"Error printing final metrics: {e}")
