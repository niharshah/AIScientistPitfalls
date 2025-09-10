import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    drop_dict = experiment_data.get("dropout_tuning", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    drop_dict = {}

# Figure 1: Loss curves (train & val) for all dropout rates
try:
    plt.figure(figsize=(8, 5))
    for dr_str, rec in drop_dict.items():
        epochs = np.arange(1, len(rec["losses"]["train"]) + 1)
        plt.plot(epochs, rec["losses"]["train"], "--", label=f"{dr_str} train")
        plt.plot(epochs, rec["losses"]["val"], "-", label=f"{dr_str} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Synthetic Dataset — Loss Curves\nLeft: Training, Right: Validation")
    plt.legend(fontsize=7, ncol=2)
    fname = os.path.join(working_dir, "synthetic_loss_curves_all_dropout.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Figure 2: Validation accuracy curves for all dropout rates
try:
    plt.figure(figsize=(8, 5))
    for dr_str, rec in drop_dict.items():
        accs = [m["acc"] for m in rec["metrics"]["val"]]
        epochs = np.arange(1, len(accs) + 1)
        plt.plot(epochs, accs, marker="o", label=dr_str)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1.05)
    plt.title("Synthetic Dataset — Validation Accuracy vs. Epoch")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "synthetic_val_accuracy_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# Figure 3: Final validation accuracy bar chart per dropout
try:
    plt.figure(figsize=(6, 4))
    drs, finals = [], []
    for dr_str, rec in drop_dict.items():
        drs.append(dr_str.replace("dropout_", ""))
        finals.append(rec["metrics"]["val"][-1]["acc"])
    plt.bar(drs, finals, color="skyblue")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Final Validation Accuracy")
    for i, v in enumerate(finals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    plt.ylim(0, 1.05)
    plt.title("Synthetic Dataset — Final Epoch Validation Accuracies")
    fname = os.path.join(working_dir, "synthetic_final_val_accuracy_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()
