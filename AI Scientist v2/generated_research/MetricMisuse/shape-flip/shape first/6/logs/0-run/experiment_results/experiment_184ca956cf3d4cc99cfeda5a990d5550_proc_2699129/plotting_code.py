import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    losses_tr = data["losses"]["train"]
    losses_val = data["losses"]["val"]
    val_metrics = data["metrics"]["val"]  # list of dicts
    epochs = list(range(1, len(losses_tr) + 1))

    # Helper to extract metric series safely
    def metric_series(key):
        return [m.get(key, np.nan) for m in val_metrics]

    # 1) Loss curves ------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train Loss")
        plt.plot(epochs, losses_val, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Accuracy curve --------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, metric_series("acc"), marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Validation Accuracy per Epoch")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3) Shape-weighted accuracy -----------------------------------
    try:
        plt.figure()
        plt.plot(epochs, metric_series("swa"), marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Acc")
        plt.title("SPR_BENCH: Validation SWA per Epoch")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_SWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 4) Color-weighted accuracy -----------------------------------
    try:
        plt.figure()
        plt.plot(epochs, metric_series("cwa"), marker="o", color="magenta")
        plt.xlabel("Epoch")
        plt.ylabel("Color-Weighted Acc")
        plt.title("SPR_BENCH: Validation CWA per Epoch")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_CWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Print final test metrics
    test_metrics = data["metrics"]["test"]
    print("\nTest set metrics:")
    for k, v in test_metrics.items():
        print(f"  {k.upper():4s}: {v:.3f}")
