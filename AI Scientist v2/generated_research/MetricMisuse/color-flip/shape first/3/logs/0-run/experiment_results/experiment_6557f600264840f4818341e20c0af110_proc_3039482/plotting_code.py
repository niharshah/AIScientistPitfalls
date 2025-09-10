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
    ed = experiment_data["no_pretraining"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = range(1, len(ed["losses"]["train"]) + 1)

    # -----------------------------#
    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -----------------------------#
    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, ed["metrics"]["val_acc"], label="Val Acc")
        plt.plot(epochs, ed["metrics"]["val_aca"], label="Val ACA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Accuracy & ACA over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # -----------------------------#
    # 3) Test metrics bar chart
    try:
        test_metrics = ed["test"]
        names = ["Acc", "SWA", "CWA", "ACA"]
        values = [
            test_metrics.get("acc", 0),
            test_metrics.get("swa", 0),
            test_metrics.get("cwa", 0),
            test_metrics.get("aca", 0),
        ]
        plt.figure()
        plt.bar(names, values, color=["steelblue", "orange", "green", "red"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test Metrics")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()
