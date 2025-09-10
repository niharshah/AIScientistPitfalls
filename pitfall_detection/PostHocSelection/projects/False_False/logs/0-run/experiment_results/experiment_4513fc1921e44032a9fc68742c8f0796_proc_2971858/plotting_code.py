import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["frozen_encoder"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    # ------------------------------------------------------------------
    # Plot 1: Loss curves
    try:
        plt.figure()
        if exp["losses"]["pretrain"]:
            plt.plot(exp["losses"]["pretrain"], label="Pre-train loss")
        if exp["losses"]["train"]:
            plt.plot(
                range(
                    len(exp["losses"]["pretrain"]),
                    len(exp["losses"]["pretrain"]) + len(exp["losses"]["train"]),
                ),
                exp["losses"]["train"],
                label="Train loss",
            )
        if exp["losses"]["val"]:
            plt.plot(
                range(
                    len(exp["losses"]["pretrain"]),
                    len(exp["losses"]["pretrain"]) + len(exp["losses"]["val"]),
                ),
                exp["losses"]["val"],
                label="Val loss",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Pre-train / Train / Val Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Validation metric curves
    try:
        plt.figure()
        if exp["metrics"]["val_SWA"]:
            plt.plot(exp["metrics"]["val_SWA"], label="SWA")
        if exp["metrics"]["val_CWA"]:
            plt.plot(exp["metrics"]["val_CWA"], label="CWA")
        if exp["metrics"]["val_SCWA"]:
            plt.plot(exp["metrics"]["val_SCWA"], label="SCWA")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Validation Metrics")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Print best metrics
    if exp["metrics"]["val_SCWA"]:
        best_idx = int(np.argmax(exp["metrics"]["val_SCWA"]))
        best_swa = exp["metrics"]["val_SWA"][best_idx]
        best_cwa = exp["metrics"]["val_CWA"][best_idx]
        best_scwa = exp["metrics"]["val_SCWA"][best_idx]
        print(
            f"Best epoch: {best_idx + 1} | SWA={best_swa:.4f} "
            f"CWA={best_cwa:.4f} SCWA={best_scwa:.4f}"
        )
    else:
        print("No validation metrics recorded.")
