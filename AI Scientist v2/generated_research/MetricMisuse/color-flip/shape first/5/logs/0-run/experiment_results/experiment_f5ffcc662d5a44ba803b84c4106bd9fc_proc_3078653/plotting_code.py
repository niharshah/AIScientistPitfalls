import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- setup & load -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
ds = experiment_data.get(ds_name, {})
losses = ds.get("losses", {})
metrics = ds.get("metrics", {})
preds = np.array(ds.get("predictions", []))
truths = np.array(ds.get("ground_truth", []))

# x-axis (epochs)
epochs = np.arange(1, len(losses.get("train", [])) + 1)

# ----------- plot 1: loss ------------
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train")
    plt.plot(epochs, losses.get("val", []), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{ds_name} – Training & Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ----------- plot 2: SWA -------------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train_SWA", []), label="Train SWA")
    plt.plot(epochs, metrics.get("val_SWA", []), label="Val SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title(f"{ds_name} – SWA Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_SWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ----------- plot 3: CWA & ACR -------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train_CWA", []), label="Train CWA")
    plt.plot(epochs, metrics.get("val_CWA", []), label="Val CWA")
    plt.plot(epochs, metrics.get("train_ACR", []), "--", label="Train ACR")
    plt.plot(epochs, metrics.get("val_ACR", []), "--", label="Val ACR")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"{ds_name} – CWA & ACR Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_CWA_ACR_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA/ACR plot: {e}")
    plt.close()

# ----------- plot 4: confusion matrix-
try:
    if preds.size and truths.size:
        num_labels = int(max(truths.max(), preds.max()) + 1)
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(truths, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_name} – Confusion Matrix (Test)")
        for i in range(num_labels):
            for j in range(num_labels):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ----------- print final test metrics -
try:
    print(
        "Final Test Metrics:",
        "SWA=",
        metrics.get("test_SWA", [-1])[-1],
        "CWA=",
        metrics.get("test_CWA", [-1])[-1],
        "ACR=",
        metrics.get("test_ACR", [-1])[-1],
    )
except Exception as e:
    print(f"Error printing metrics: {e}")
