import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data["sequential_only"]["SPR_BENCH"]
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    val_metrics = data["metrics"]["val"]  # list of dicts per epoch
    test_metrics = data["metrics"]["test"]  # dict with cwa, swa, cpxwa

    epochs = range(1, len(train_losses) + 1)
    cwa_vals = [m["cwa"] for m in val_metrics]
    swa_vals = [m["swa"] for m in val_metrics]
    cpx_vals = [m["cpxwa"] for m in val_metrics]

    # ------------- Plot 1: Loss curves -----------------
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------- Plot 2: Validation metrics ----------
    try:
        plt.figure()
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.plot(epochs, swa_vals, label="SWA")
        plt.plot(epochs, cpx_vals, label="CpxWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Validation Metrics Across Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation metrics plot: {e}")
        plt.close()

    # ------------- Plot 3: Test metrics ----------------
    try:
        plt.figure()
        names = ["CWA", "SWA", "CpxWA"]
        vals = [
            test_metrics.get("cwa", 0),
            test_metrics.get("swa", 0),
            test_metrics.get("cpxwa", 0),
        ]
        plt.bar(names, vals, color=["skyblue", "salmon", "lightgreen"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Test Metrics (Weighted Accuracies)")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()
