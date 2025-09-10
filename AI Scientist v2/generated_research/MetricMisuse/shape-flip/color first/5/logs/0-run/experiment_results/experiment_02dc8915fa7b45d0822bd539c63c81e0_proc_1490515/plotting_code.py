import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- Load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["hidden_dim"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

if runs:  # proceed only if data exists
    hds = sorted(int(k.split("_")[1]) for k in runs.keys())
    # Collect per-hd arrays
    loss_dict, acc_dict, test_acc = {}, {}, {}
    for hd in hds:
        rd = runs[f"hd_{hd}"]
        loss_dict[hd] = (rd["losses"]["train"], rd["losses"]["val"])
        acc_dict[hd] = (rd["metrics"]["train"], rd["metrics"]["val"])
        test_acc[hd] = rd["test"]["CpxWA"]

    # ---------------- Figure 1: Loss curves ----------------
    try:
        plt.figure(figsize=(6, 4))
        for hd in hds:
            epochs = range(1, len(loss_dict[hd][0]) + 1)
            plt.plot(epochs, loss_dict[hd][0], label=f"train hd={hd}", linestyle="-")
            plt.plot(epochs, loss_dict[hd][1], label=f"val hd={hd}", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset: Training vs. Validation Loss by Hidden Dim")
        plt.legend(fontsize="small")
        fname = os.path.join(working_dir, "SPR_loss_curves_hidden_dim.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------------- Figure 2: Accuracy curves ----------------
    try:
        plt.figure(figsize=(6, 4))
        for hd in hds:
            epochs = range(1, len(acc_dict[hd][0]) + 1)
            plt.plot(epochs, acc_dict[hd][0], label=f"train hd={hd}", linestyle="-")
            plt.plot(epochs, acc_dict[hd][1], label=f"val hd={hd}", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("SPR Dataset: Training vs. Validation CpxWA by Hidden Dim")
        plt.legend(fontsize="small")
        fname = os.path.join(working_dir, "SPR_accuracy_curves_hidden_dim.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------------- Figure 3: Test accuracy bar chart ----------------
    try:
        plt.figure(figsize=(5, 3))
        plt.bar(range(len(hds)), [test_acc[hd] for hd in hds], tick_label=hds)
        plt.ylabel("Test CpxWA")
        plt.xlabel("Hidden Dimension")
        plt.title("SPR Dataset: Test CpxWA vs. Hidden Dim")
        fname = os.path.join(working_dir, "SPR_test_CpxWA_bar_hidden_dim.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating test bar plot: {e}")
        plt.close()

    # ---------------- Print best configs ----------------
    best_val = max(hds, key=lambda hd: acc_dict[hd][1][-1])
    best_test = max(hds, key=lambda hd: test_acc[hd])
    print(
        f"Best hidden_dim by final validation CpxWA: {best_val} "
        f"(val={acc_dict[best_val][1][-1]:.4f})"
    )
    print(
        f"Best hidden_dim by test CpxWA: {best_test} "
        f"(test={test_acc[best_test]:.4f})"
    )
else:
    print("No runs found in experiment_data.npy")
