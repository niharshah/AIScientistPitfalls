import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data -------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["single_gat_layer"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    epochs = np.array(exp["epochs"])
    train_loss = np.array(exp["losses"]["train"])
    val_loss = np.array(exp["losses"]["val"])
    train_hwa = np.array([m["HWA"] for m in exp["metrics"]["train"]])
    val_hwa = np.array([m["HWA"] for m in exp["metrics"]["val"]])
    test_met = {
        k: v for k, v in zip(["CWA", "SWA", "HWA"], exp["metrics"]["val"][-1].values())
    }

    # -------------------- loss curve -------------------- #
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR: Train vs Val Loss (1-Hop GAT)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------------------- HWA curve -------------------- #
    try:
        plt.figure()
        plt.plot(epochs, train_hwa, label="Train")
        plt.plot(epochs, val_hwa, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy (HWA)")
        plt.title("SPR: Train vs Val HWA (1-Hop GAT)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_hwa_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve: {e}")
        plt.close()

    # -------------------- test bar chart -------------------- #
    try:
        plt.figure()
        bars = ["CWA", "SWA", "HWA"]
        vals = [test_met["CWA"], test_met["SWA"], test_met["HWA"]]
        plt.bar(bars, vals, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.title("SPR Test Metrics (1-Hop GAT)\nLeft: CWA, Center: SWA, Right: HWA")
        plt.savefig(os.path.join(working_dir, "SPR_test_metrics.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar chart: {e}")
        plt.close()

    # -------------------- print final metrics -------------------- #
    print(f"Test metrics -> CWA: {vals[0]:.3f}, SWA: {vals[1]:.3f}, HWA: {vals[2]:.3f}")
