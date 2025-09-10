import matplotlib.pyplot as plt
import numpy as np
import os

# House-keeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Proceed only if expected dataset is present
if "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    metrics = data["metrics"]
    train_loss = metrics.get("train_loss", [])
    val_loss = metrics.get("val_loss", [])
    swa = metrics.get("SWA", [])
    cwa = metrics.get("CWA", [])
    hwa = metrics.get("HWA", [])
    preds = data.get("predictions", [])
    gts = data.get("ground_truth", [])
    epochs = list(range(1, len(train_loss) + 1))

    # Plot 1 ─ Train / Val loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss\nLeft: Train, Right: Val")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # Plot 2 ─ Weighted accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, hwa, label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Weighted Accuracies Across Epochs\nSWA, CWA, HWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_weighted_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating weighted accuracy plot: {e}")
        plt.close()

    # Plot 3 ─ Confusion matrix (final epoch)
    try:
        if preds and gts:
            labels = sorted(list(set(gts) | set(preds)))
            label_idx = {lab: i for i, lab in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[label_idx[t], label_idx[p]] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR_BENCH – Confusion Matrix\nLeft: GT, Right: Pred")
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # Print final epoch evaluation metrics
    if swa and cwa and hwa:
        print(
            f"Final Epoch Metrics -> SWA: {swa[-1]:.3f}, CWA: {cwa[-1]:.3f}, HWA: {hwa[-1]:.3f}"
        )
else:
    print("SPR_BENCH data not found in experiment_data.")
