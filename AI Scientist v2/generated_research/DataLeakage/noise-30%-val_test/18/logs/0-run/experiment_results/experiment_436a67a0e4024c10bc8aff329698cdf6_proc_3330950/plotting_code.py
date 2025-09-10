import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup & data loading ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    lr_runs = experiment_data.get("learning_rate", {})
    lrs = sorted(lr_runs.keys(), key=lambda x: float(x.split("_")[1]))
    dataset_name = "SPR_BENCH"  # fallback name
    epochs = lr_runs[lrs[0]]["epochs"] if lrs else []

    # -------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        for lr_key in lrs:
            d = lr_runs[lr_key]["losses"]
            plt.plot(epochs, d["train"], "--", label=f"{lr_key} train")
            plt.plot(epochs, d["val"], "-", label=f"{lr_key} val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset_name} Loss Curves\nLeft: dashed=train, solid=val")
        plt.legend(fontsize="small")
        fname = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- Plot 2: F1 curves ------------
    try:
        plt.figure()
        for lr_key in lrs:
            m = lr_runs[lr_key]["metrics"]
            plt.plot(epochs, m["train"], "--", label=f"{lr_key} train")
            plt.plot(epochs, m["val"], "-", label=f"{lr_key} val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dataset_name} Macro-F1 Curves\nLeft: dashed=train, solid=val")
        plt.legend(fontsize="small")
        fname = os.path.join(working_dir, f"{dataset_name}_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # -------- Plot 3: Dev/Test bar chart ---
    try:
        best_dev = [lr_runs[k]["best_dev_f1"] for k in lrs]
        test_f1 = [lr_runs[k]["test_f1"] for k in lrs]
        x = np.arange(len(lrs))
        width = 0.35

        plt.figure()
        plt.bar(x - width / 2, best_dev, width, label="Best Dev F1")
        plt.bar(x + width / 2, test_f1, width, label="Test F1")
        plt.xticks(x, lrs, rotation=45)
        plt.ylabel("Macro-F1")
        plt.title(f"{dataset_name} Dev vs Test F1 per Learning Rate")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_name}_dev_test_f1_bar.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()

    # -------- Print numeric summary -------
    print("Learning Rate | Best Dev F1 | Test F1")
    for lr_key in lrs:
        print(
            f"{lr_key:<13} {lr_runs[lr_key]['best_dev_f1']:.4f}      {lr_runs[lr_key]['test_f1']:.4f}"
        )
