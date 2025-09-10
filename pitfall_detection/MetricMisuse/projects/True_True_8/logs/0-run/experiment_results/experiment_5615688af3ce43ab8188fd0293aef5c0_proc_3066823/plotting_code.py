import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- plotting ----------
for dname, entry in experiment_data.items():
    epochs = range(1, len(entry["losses"]["train"]) + 1)

    # 1. loss curves
    try:
        plt.figure()
        plt.plot(epochs, entry["losses"]["train"], label="Train Loss")
        plt.plot(epochs, entry["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} Loss Curves\nTraining vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # 2. accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, entry["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, entry["metrics"]["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dname} Accuracy Curves\nTraining vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
        plt.close()

    # 3. augmentation-consistency score
    try:
        plt.figure()
        plt.plot(epochs, entry["metrics"]["val_acs"], label="Val ACS")
        plt.xlabel("Epoch")
        plt.ylabel("Aug. Consistency")
        plt.title(f"{dname} Validation Augmentation Consistency Score")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_acs_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating ACS plot for {dname}: {e}")
        plt.close()

    # ---------- summary print ----------
    if entry["metrics"]["val_acc"]:
        print(f"{dname} final Val Acc: {entry['metrics']['val_acc'][-1]:.4f}")
    if entry["metrics"]["val_acs"]:
        print(f"{dname} final Val ACS: {entry['metrics']['val_acs'][-1]:.4f}")
