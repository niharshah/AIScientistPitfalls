import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    spr = experiment_data["SPR_BENCH"]

    # Extract losses & metrics
    epochs = [e for e, _ in spr["losses"]["train"]]
    tr_loss = [v for _, v in spr["losses"]["train"]]
    val_loss = [v for _, v in spr["losses"]["val"]]
    tr_swa = [v for _, v in spr["metrics"]["train"]]
    val_swa = [v for _, v in spr["metrics"]["val"]]

    # Test accuracy
    gts = spr.get("ground_truth", [])
    preds = spr.get("predictions", [])
    test_acc = (
        (sum(int(g == p) for g, p in zip(gts, preds)) / len(gts)) if gts else np.nan
    )
    best_val_swa = max(val_swa) if val_swa else np.nan

    # -------------------------------------------------------------- #
    # Plot 1 : Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title("SPR_BENCH – Cross-Entropy Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # Plot 2 : SWA curves
    try:
        plt.figure()
        plt.plot(epochs, tr_swa, label="Train")
        plt.plot(epochs, val_swa, label="Validation")
        plt.title("SPR_BENCH – Shape-Weighted Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # Plot 3 : Test accuracy bar chart
    try:
        plt.figure()
        plt.bar(["SPR_BENCH"], [test_acc])
        plt.title("SPR_BENCH – Test Accuracy")
        plt.ylabel("Accuracy")
        for i, a in enumerate([test_acc]):
            plt.text(i, a + 0.01, f"{a:.2f}", ha="center", va="bottom")
        fname = os.path.join(working_dir, "spr_bench_test_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy chart: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # Print metrics
    print(f"Best Validation SWA: {best_val_swa:.4f}")
    print(f"Test Accuracy      : {test_acc:.4f}")
else:
    print("No SPR_BENCH data found.")
