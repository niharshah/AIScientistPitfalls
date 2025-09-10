import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "SPR_BENCH"
keys = sorted(experiment_data.keys())
epochs = len(next(iter(experiment_data.values()))[dataset_name]["losses"]["train"])
epoch_axis = np.arange(1, epochs + 1)

# gather results
final_f1 = {}
for k in keys:
    f1_vals = experiment_data[k][dataset_name]["metrics"]["val"]
    if len(f1_vals):
        final_f1[k] = f1_vals[-1]

best_key = max(final_f1, key=final_f1.get)

# -------------------------------------------------
# 1) Training & Validation Loss plot
try:
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for k in keys:
        train_losses = experiment_data[k][dataset_name]["losses"]["train"]
        val_losses = experiment_data[k][dataset_name]["losses"]["val"]
        ax[0].plot(epoch_axis, train_losses, label=k)
        ax[1].plot(epoch_axis, val_losses, label=k)
    ax[0].set_ylabel("Train Loss")
    ax[1].set_ylabel("Val Loss")
    ax[1].set_xlabel("Epoch")
    ax[0].set_title(f"{dataset_name} - Loss Curves (Top: Train, Bottom: Val)")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# -------------------------------------------------
# 2) Validation Macro-F1 curves
try:
    plt.figure(figsize=(8, 5))
    for k in keys:
        f1_vals = experiment_data[k][dataset_name]["metrics"]["val"]
        plt.plot(epoch_axis, f1_vals, label=k)
    plt.title(f"{dataset_name} - Validation Macro-F1 over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_val_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 curve plot: {e}")
    plt.close()

# -------------------------------------------------
# 3) Final Macro-F1 bar chart
try:
    plt.figure(figsize=(8, 5))
    plt.bar(list(final_f1.keys()), list(final_f1.values()))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final Macro-F1")
    plt.title(f"{dataset_name} - Final Epoch Macro-F1 per Weight Decay")
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_final_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# -------------------------------------------------
# 4) Confusion matrix for best config
try:
    preds = experiment_data[best_key][dataset_name]["predictions"]
    gts = experiment_data[best_key][dataset_name]["ground_truth"]
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(
        f"{dataset_name} - Confusion Matrix (Best: {best_key})\nLeft: Ground Truth, Right: Predicted"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_confusion_matrix_{best_key}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------------------------------------------------
# Print final metrics
print("Final Macro-F1 per weight_decay:")
for k, v in final_f1.items():
    print(f"{k}: {v:.4f}")
