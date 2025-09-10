import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
# Load stored experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

dataset_names = list(experiment_data.keys())
cross_val_cpx = {}

# ---------------------------------------------------------------
for dname in dataset_names:
    ed = experiment_data[dname]
    epochs = np.array(ed.get("epochs", []))
    if epochs.size == 0:  # skip empty entries
        continue
    train_losses = np.array(ed["losses"]["train"])
    train_metrics = ed["metrics"]["train"]
    val_metrics = ed["metrics"]["val"]

    train_cpx = np.array([m["cpx"] for m in train_metrics])
    val_cpx = np.array([m["cpx"] for m in val_metrics])
    val_cwa = np.array([m["cwa"] for m in val_metrics])
    val_swa = np.array([m["swa"] for m in val_metrics])
    cross_val_cpx[dname] = (epochs, val_cpx)

    # 1) Training loss curve
    try:
        plt.figure()
        plt.plot(epochs, train_losses, marker="o", label="Train Loss")
        plt.title(f"{dname}: Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_train_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # 2) Train vs. Validation Complexity-Weighted Accuracy
    try:
        plt.figure()
        plt.plot(epochs, train_cpx, marker="o", label="Train CpxWA")
        plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
        plt.title(
            f"{dname}: Complexity-Weighted Accuracy\nLeft: Train, Right: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_cpxwa_train_val_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CpxWA curve for {dname}: {e}")
        plt.close()

    # 3) Validation weighted-accuracy comparison
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, marker="o", label="Val CWA")
        plt.plot(epochs, val_swa, marker="^", label="Val SWA")
        plt.plot(epochs, val_cpx, marker="s", label="Val CpxWA")
        plt.title(f"{dname}: Weighted Accuracy Comparison (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dname}_val_weighted_accuracy_comparison.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating weighted accuracy plot for {dname}: {e}")
        plt.close()

# ---------------------------------------------------------------
# 4) Cross-dataset comparison of validation CpxWA (only if >1 dataset)
if len(cross_val_cpx) > 1:
    try:
        plt.figure()
        for dname, (ep, vcpx) in cross_val_cpx.items():
            plt.plot(ep, vcpx, marker="o", label=f"{dname}")
        plt.title("Validation Complexity-Weighted Accuracy Across Datasets")
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "cross_dataset_val_cpxwa_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset plot: {e}")
        plt.close()
