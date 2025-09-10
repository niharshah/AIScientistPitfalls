import matplotlib.pyplot as plt
import numpy as np
import os

# working directory for saving plots
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------
# Load experiment data ------------------------------------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
data = experiment_data.get(dataset, {})
epochs = data.get("epochs", [])
losses = data.get("losses", {})
metrics = data.get("metrics", {})
preds = np.array(data.get("predictions", []))
gts = np.array(data.get("ground_truth", []))

# -------------------------------------------------------------------
# 1) Train / Val Loss Curves ----------------------------------------
try:
    if epochs and losses:
        plt.figure()
        plt.plot(epochs, losses.get("train", []), label="Train Loss")
        plt.plot(epochs, losses.get("val", []), label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset} Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------------------------------------------------------
# 2) Validation SCWA over Epochs ------------------------------------
try:
    if epochs and metrics and metrics.get("val"):
        plt.figure()
        plt.plot(epochs, metrics["val"], marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("SCWA")
        plt.title(f"{dataset} Validation SCWA")
        fname = os.path.join(working_dir, f"{dataset}_val_SCWA.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating SCWA plot: {e}")
    plt.close()

# -------------------------------------------------------------------
# 3) Confusion Matrix Heat-map --------------------------------------
try:
    if preds.size and gts.size:
        num_classes = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dataset} Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        fname = os.path.join(working_dir, f"{dataset}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# -------------------------------------------------------------------
# Print simple evaluation summaries ---------------------------------
try:
    final_scwa = metrics.get("val", [None])[-1]
    accuracy = (preds == gts).mean() if preds.size else None
    print(f"Final Validation SCWA: {final_scwa}")
    print(f"Test Accuracy: {accuracy}")
except Exception as e:
    print(f"Error computing summary metrics: {e}")
