import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------------------------#
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    # Convenient handles
    model_key = "NoRecurrentEncoder"
    ds_key = "SPR_BENCH"
    try:
        data = experiment_data[model_key][ds_key]
        losses = data["losses"]
        metrics = data["metrics"]
        test = data["test"]
    except Exception as e:
        print(f"Error extracting keys: {e}")
        data = None

    # -------------------------------------------------------------------------#
    # 1) Loss curves
    try:
        plt.figure()
        x_pre = range(1, len(losses["pretrain"]) + 1)
        x_ft = range(1, len(losses["train"]) + 1)
        plt.plot(x_pre, losses["pretrain"], label="Pretrain loss")
        plt.plot(x_ft, losses["train"], label="Train loss")
        plt.plot(x_ft, losses["val"], label="Val loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Loss Curves\nLeft: Pretraining, Right: Fine-tuning")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # -------------------------------------------------------------------------#
    # 2) Validation metrics curves
    try:
        plt.figure()
        x_ft = range(1, len(metrics["val_acc"]) + 1)
        plt.plot(x_ft, metrics["val_acc"], label="Val Accuracy")
        plt.plot(x_ft, metrics["val_aca"], label="Val ACA")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Validation Metrics Over Fine-tuning")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_validation_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation metrics plot: {e}")
        plt.close()

    # -------------------------------------------------------------------------#
    # 3) Test metrics bar chart
    try:
        plt.figure()
        names = ["Acc", "SWA", "CWA", "ACA"]
        values = [test.get("acc"), test.get("swa"), test.get("cwa"), test.get("aca")]
        plt.bar(names, values, color="skyblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test Metrics Summary")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar chart: {e}")
        plt.close()
