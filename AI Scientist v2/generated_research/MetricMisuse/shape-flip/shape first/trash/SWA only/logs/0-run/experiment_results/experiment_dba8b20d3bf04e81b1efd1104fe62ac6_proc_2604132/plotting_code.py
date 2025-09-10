import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    ed = experiment_data["SPR_BENCH"]
    epochs = range(1, len(ed["losses"]["train"]) + 1)

    # 1) Loss curves
    try:
        if ed["losses"]["train"] and ed["losses"]["val"]:
            plt.figure()
            plt.plot(epochs, ed["losses"]["train"], label="Train")
            plt.plot(epochs, ed["losses"]["val"], label="Validation")
            plt.title("SPR_BENCH – Loss vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        tr_acc = [m["acc"] for m in ed["metrics"]["train"]]
        va_acc = [m["acc"] for m in ed["metrics"]["val"]]
        if tr_acc and va_acc:
            plt.figure()
            plt.plot(epochs, tr_acc, label="Train")
            plt.plot(epochs, va_acc, label="Validation")
            plt.title("SPR_BENCH – Accuracy vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves: {e}")
        plt.close()

    # 3) Shape-Weighted Accuracy curves
    try:
        tr_swa = [m["swa"] for m in ed["metrics"]["train"]]
        va_swa = [m["swa"] for m in ed["metrics"]["val"]]
        if tr_swa and va_swa:
            plt.figure()
            plt.plot(epochs, tr_swa, label="Train")
            plt.plot(epochs, va_swa, label="Validation")
            plt.title("SPR_BENCH – Shape-Weighted Accuracy vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating SWA curves: {e}")
        plt.close()

    # 4) Test metrics bar chart
    try:
        test_acc = ed["metrics"]["test"].get("acc", None)
        test_swa = ed["metrics"]["test"].get("swa", None)
        if test_acc is not None and test_swa is not None:
            plt.figure()
            plt.bar(
                ["Accuracy", "SWA"], [test_acc, test_swa], color=["steelblue", "orange"]
            )
            plt.title("SPR_BENCH – Final Test Metrics")
            plt.ylabel("Score")
            fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
        plt.close()

    # --------------- print final numbers ------------------------------
    print(
        "Final Test Metrics – Acc: {:.3f}, SWA: {:.3f}".format(
            ed["metrics"]["test"].get("acc", float("nan")),
            ed["metrics"]["test"].get("swa", float("nan")),
        )
    )
