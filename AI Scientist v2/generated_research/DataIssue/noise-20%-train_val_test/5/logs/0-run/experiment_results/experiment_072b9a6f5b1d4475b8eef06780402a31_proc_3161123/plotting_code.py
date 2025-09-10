import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup & load -------- #
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
data_dict = experiment_data.get("num_layers", {}).get(ds_name, {})

# -------- accuracy curves -------- #
try:
    plt.figure(figsize=(8, 5))
    for key, info in sorted(data_dict.items()):
        epochs = range(1, len(info["metrics"]["train_acc"]) + 1)
        plt.plot(epochs, info["metrics"]["train_acc"], label=f"{key}_train")
        plt.plot(epochs, info["metrics"]["val_acc"], "--", label=f"{key}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{ds_name} Training / Validation Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# -------- loss curves -------- #
try:
    plt.figure(figsize=(8, 5))
    for key, info in sorted(data_dict.items()):
        epochs = range(1, len(info["losses"]["train"]) + 1)
        plt.plot(epochs, info["losses"]["train"], label=f"{key}_train")
        plt.plot(epochs, info["losses"]["val"], "--", label=f"{key}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{ds_name} Training / Validation Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------- test accuracy bar -------- #
try:
    plt.figure(figsize=(6, 4))
    layer_labels, test_accs = [], []
    for key, info in sorted(data_dict.items()):
        layer_labels.append(key.split("_")[1])  # extracts the numeric layer count
        test_accs.append(info["test_acc"])
    plt.bar(layer_labels, test_accs, color="skyblue")
    plt.xlabel("Number of Layers")
    plt.ylabel("Test Accuracy")
    plt.title(f"{ds_name} Test Accuracy vs Num Layers")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_test_accuracy_bar.png"))
    plt.close()
    for l, acc in zip(layer_labels, test_accs):
        print(f"Num layers {l}: test_acc={acc:.3f}")
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()
