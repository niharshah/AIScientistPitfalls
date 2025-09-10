import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths & data ---
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
    spr_logs = experiment_data["batch_size"]["SPR"]
    batch_sizes = sorted(int(b) for b in spr_logs.keys())

    # helper to get epochs range (all have same length)
    epochs = range(1, len(next(iter(spr_logs.values()))["losses"]["train"]) + 1)

    # 1) Loss curves ---------------------------------------------------------
    try:
        plt.figure(figsize=(7, 5))
        for bs in batch_sizes:
            log = spr_logs[str(bs)]
            plt.plot(epochs, log["losses"]["train"], label=f"Train bs{bs}")
            plt.plot(epochs, log["losses"]["val"], linestyle="--", label=f"Val bs{bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Loss Curves\nSolid: Training, Dashed: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2) Metric curves (SWA & CWA) ------------------------------------------
    try:
        plt.figure(figsize=(7, 5))
        for bs in batch_sizes:
            log = spr_logs[str(bs)]
            plt.plot(epochs, log["metrics"]["train"], label=f"SWA-Train bs{bs}")
            plt.plot(
                epochs, log["metrics"]["val"], linestyle="--", label=f"CWA-Val bs{bs}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR Accuracy Curves\nSolid: SWA (Train), Dashed: CWA (Val)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_accuracy_curves.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating metric curve plot: {e}")
        plt.close()

    # 3) AIS curves ----------------------------------------------------------
    try:
        plt.figure(figsize=(7, 5))
        for bs in batch_sizes:
            log = spr_logs[str(bs)]
            plt.plot(epochs, log["AIS"]["val"], marker="o", label=f"AIS bs{bs}")
        plt.xlabel("Epoch")
        plt.ylabel("AIS")
        plt.title("SPR AIS Consistency Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_AIS_curves.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating AIS plot: {e}")
        plt.close()

    print("Plots saved to:", working_dir)
