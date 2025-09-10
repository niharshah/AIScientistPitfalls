import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    d = experiment_data["scalar_free_symbolic"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    d = None

if d is not None:
    epochs = np.arange(1, len(d["metrics"]["train"]) + 1)

    # Accuracy plot
    try:
        plt.figure()
        plt.plot(epochs, d["metrics"]["train"], label="Train")
        plt.plot(epochs, d["metrics"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves – Scalar-Free Symbolic")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # Loss plot
    try:
        plt.figure()
        plt.plot(epochs, d["losses"]["train"], label="Train")
        plt.plot(epochs, d["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves – Scalar-Free Symbolic")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # Shape-weighted accuracy plot
    try:
        plt.figure()
        plt.plot(epochs, d["swa"]["train"], label="Train")
        plt.plot(epochs, d["swa"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH SWA Curves – Scalar-Free Symbolic")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # print final test metrics
    tm = d.get("test_metrics", {})
    if tm:
        print(
            f"Test Loss: {tm.get('loss'):.4f}  Test Acc: {tm.get('acc'):.3f}  Test SWA: {tm.get('swa'):.3f}"
        )
