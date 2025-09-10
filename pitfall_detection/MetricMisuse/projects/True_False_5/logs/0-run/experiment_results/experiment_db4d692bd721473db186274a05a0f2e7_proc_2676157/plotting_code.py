import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    lr_dict = experiment_data.get("learning_rate_tuning", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    lr_dict = {}


# helper to get sorted keys numerically
def sort_lr(keys):
    def _num(k):
        try:
            return float(k.split("_")[1])
        except Exception:
            return np.inf

    return sorted(keys, key=_num)


# ------------------ plot 1: loss curves ------------------
try:
    if lr_dict:
        plt.figure(figsize=(6, 4))
        for lr_key in sort_lr(lr_dict):
            d = lr_dict[lr_key]
            epochs = np.arange(1, len(d["losses"]["train"]) + 1)
            plt.plot(epochs, d["losses"]["train"], "--", label=f"{lr_key} train")
            plt.plot(epochs, d["losses"]["val"], "-", label=f"{lr_key} val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Learning-Rate Tuning: Train vs Val Loss")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "learning_rate_tuning_loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------ plot 2: RCWA curves ------------------
try:
    if lr_dict:
        plt.figure(figsize=(6, 4))
        for lr_key in sort_lr(lr_dict):
            d = lr_dict[lr_key]
            epochs = np.arange(1, len(d["metrics"]["val_rcwa"]) + 1)
            plt.plot(epochs, d["metrics"]["val_rcwa"], marker="o", label=lr_key)
        plt.xlabel("Epoch")
        plt.ylabel("Validation RCWA")
        plt.title("Learning-Rate Tuning: Validation RCWA per Epoch")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "learning_rate_tuning_rcwa_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating RCWA curves: {e}")
    plt.close()

# ------------------ plot 3: test RCWA bar ------------------
try:
    if lr_dict:
        lr_keys = sort_lr(lr_dict)
        rcwas = [lr_dict[k]["test_metrics"]["RCWA"] for k in lr_keys]
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(len(lr_keys)), rcwas, tick_label=lr_keys)
        plt.ylabel("Test RCWA")
        plt.title("Learning-Rate Tuning: Test RCWA Comparison")
        fname = os.path.join(working_dir, "learning_rate_tuning_test_rcwa.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test RCWA bar: {e}")
    plt.close()
