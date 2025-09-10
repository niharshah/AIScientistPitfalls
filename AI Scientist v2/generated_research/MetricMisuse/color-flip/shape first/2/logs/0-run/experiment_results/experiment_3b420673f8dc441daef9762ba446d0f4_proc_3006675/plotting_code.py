import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# convenient pointer
spr_exp = experiment_data.get("unidirectional_lstm", {}).get("spr_bench", {})

# -------------------------------------------------
# 1. Loss curves
try:
    train_losses = spr_exp["losses"]["train"]
    val_losses = spr_exp["losses"]["val"]
    if len(train_losses) and len(val_losses):
        epochs = np.arange(1, len(train_losses) + 1)
        plt.figure()
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("spr_bench – Loss Curves\nLeft: Training, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------------------------------------
# 2. Metric curves
try:
    val_metrics = spr_exp["metrics"]["val"]  # list of dicts
    if val_metrics:
        epochs = [m["epoch"] for m in val_metrics]
        swa = [m["swa"] for m in val_metrics]
        cwa = [m["cwa"] for m in val_metrics]
        ccwa = [m["ccwa"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, ccwa, label="CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("spr_bench – Validation Metrics\nSWA, CWA, CCWA over epochs")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_metric_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

# -------------------------------------------------
# 3. Confusion matrix heat-map (final / best epoch)
try:
    preds = np.array(spr_exp["predictions"])
    trues = np.array(spr_exp["ground_truth"])
    if preds.size and trues.size:
        num_labels = int(max(preds.max(), trues.max())) + 1
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("spr_bench – Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
