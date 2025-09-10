import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
data = experiment_data.get(dataset, {})

loss_tr = data.get("losses", {}).get("train", [])
loss_val = data.get("losses", {}).get("val", [])
cawa_tr = data.get("metrics", {}).get("train_CAWA", [])
cawa_val = data.get("metrics", {}).get("val_CAWA", [])
preds_all = data.get("predictions", [])
gts_all = data.get("ground_truth", [])

# ------------------------------------------------------------------ #
# print evaluation metrics
if cawa_val:
    print(f"Final Train CAWA: {cawa_tr[-1]:.4f}")
    print(f"Final Val   CAWA: {cawa_val[-1]:.4f}")
    print(f"Best  Val   CAWA: {max(cawa_val):.4f}")

# ------------------------------------------------------------------ #
# Plot 1: Loss curves
try:
    plt.figure()
    epochs = range(1, len(loss_tr) + 1)
    plt.plot(epochs, loss_tr, label="Train Loss")
    plt.plot(epochs, loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset} Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Plot 2: CAWA curves
try:
    plt.figure()
    epochs = range(1, len(cawa_tr) + 1)
    plt.plot(epochs, cawa_tr, label="Train CAWA")
    plt.plot(epochs, cawa_val, label="Validation CAWA")
    plt.xlabel("Epoch")
    plt.ylabel("CAWA")
    plt.title(f"{dataset} CAWA Curves")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_CAWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CAWA curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Plot 3: Confusion matrix for last epoch (if data present)
try:
    if preds_all and gts_all:
        preds = np.array(preds_all[-1])
        gts = np.array(gts_all[-1])
        num_cls = max(gts.max(), preds.max()) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{dataset} Confusion Matrix (Last Epoch)")
        plt.xticks(range(num_cls))
        plt.yticks(range(num_cls))
        fname = os.path.join(working_dir, f"{dataset}_confusion_matrix_last_epoch.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
