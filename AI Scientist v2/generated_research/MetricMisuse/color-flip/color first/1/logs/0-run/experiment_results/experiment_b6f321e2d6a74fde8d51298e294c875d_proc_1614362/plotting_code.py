import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["no_shape_embedding"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    epochs = np.arange(1, len(exp["losses"]["train"]) + 1)

    # 1) Train / Val loss curve
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train")
        plt.plot(epochs, exp["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR – Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) CVA over epochs
    try:
        cva_vals = [m["cva"] for m in exp["metrics"]["val"]]
        test_cva = exp["metrics"]["test"]["cva"]
        plt.figure()
        plt.plot(epochs, cva_vals, marker="o", label="Validation CVA")
        plt.axhline(
            test_cva, color="r", linestyle="--", label=f"Test CVA = {test_cva:.3f}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Composite Variety Accuracy")
        plt.title("SPR – CVA Progress Across Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_CVA_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CVA curve: {e}")
        plt.close()

    # 3) Test metrics bar plot
    try:
        test_metrics = exp["metrics"]["test"]
        names = ["CWA", "SWA", "CVA"]
        vals = [test_metrics["cwa"], test_metrics["swa"], test_metrics["cva"]]
        plt.figure()
        plt.bar(names, vals, color=["steelblue", "orange", "green"])
        plt.ylim(0, 1)
        plt.title("SPR – Test Set Weighted Accuracies")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.savefig(os.path.join(working_dir, "SPR_test_metrics_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
        plt.close()
