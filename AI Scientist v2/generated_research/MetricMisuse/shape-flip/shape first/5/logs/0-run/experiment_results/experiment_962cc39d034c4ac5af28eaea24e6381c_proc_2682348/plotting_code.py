import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------------- #
# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()["multi_dataset"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    (exp,) = [{}]


# Helper: epochs
def epochs(arr):
    return list(range(1, len(arr) + 1))


# ------------------------------------------------------------------------- #
# 1) Loss curves (train / val) for all datasets in one figure  -------------
try:
    plt.figure(figsize=(6, 4))
    for name, rec in exp.items():
        plt.plot(
            epochs(rec["losses"]["train"]),
            rec["losses"]["train"],
            label=f"{name} train",
            lw=1.5,
        )
        plt.plot(
            epochs(rec["losses"]["val"]),
            rec["losses"]["val"],
            label=f"{name} val",
            ls="--",
            lw=1.5,
        )
    plt.title("Training and Validation Loss Curves across Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize=7)
    outfile = os.path.join(working_dir, "multi_dataset_loss_curves.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------------- #
# 2) SWA curves (train / val) for all datasets in one figure  --------------
try:
    plt.figure(figsize=(6, 4))
    for name, rec in exp.items():
        plt.plot(
            epochs(rec["metrics"]["train_swa"]),
            rec["metrics"]["train_swa"],
            label=f"{name} train",
            lw=1.5,
        )
        plt.plot(
            epochs(rec["metrics"]["val_swa"]),
            rec["metrics"]["val_swa"],
            label=f"{name} val",
            ls="--",
            lw=1.5,
        )
    plt.title("Training and Validation Shape-Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.legend(fontsize=7)
    outfile = os.path.join(working_dir, "multi_dataset_swa_curves.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------------- #
# 3) Test accuracy bar plot -------------------------------------------------
try:
    names, accs = [], []
    for name, rec in exp.items():
        p, g = rec.get("predictions"), rec.get("ground_truth")
        if p is None or g is None or len(p) == 0:
            continue
        accs.append((p == g).mean())
        names.append(name)
    if names:
        plt.figure(figsize=(5, 3))
        plt.bar(names, accs, color="skyblue")
        plt.ylim(0, 1)
        plt.title("Test Accuracy per Dataset")
        plt.ylabel("Accuracy")
        outfile = os.path.join(working_dir, "test_accuracy_bar.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Saved {outfile}")
        plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()
