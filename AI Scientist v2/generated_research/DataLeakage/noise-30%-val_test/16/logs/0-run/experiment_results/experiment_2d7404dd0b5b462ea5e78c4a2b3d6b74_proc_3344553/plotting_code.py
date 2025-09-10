import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ---------------- #
try:
    exp_path = os.path.join(os.getcwd(), "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# guard for missing data
def get_exp():
    try:
        return experiment_data["RawCount_NoProj"]["SPR_BENCH"]
    except Exception as e:
        print(f"Data structure missing: {e}")
        return None


exp = get_exp()
if exp:
    epochs = exp.get("epochs", [])
    losses_tr = exp.get("losses", {}).get("train", [])
    losses_val = exp.get("losses", {}).get("val", [])
    acc_tr = [m["acc"] for m in exp["metrics"]["train"]]
    acc_val = [m["acc"] for m in exp["metrics"]["val"]]
    mcc_tr = [m["MCC"] for m in exp["metrics"]["train"]]
    mcc_val = [m["MCC"] for m in exp["metrics"]["val"]]
    rma_tr = [m["RMA"] for m in exp["metrics"]["train"]]
    rma_val = [m["RMA"] for m in exp["metrics"]["val"]]
    test_metrics = exp.get("test_metrics", {})

    # -------- 1) Loss curve -------- #
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fn = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------- 2) Accuracy curve -------- #
    try:
        plt.figure()
        plt.plot(epochs, acc_tr, label="Train")
        plt.plot(epochs, acc_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Training vs Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        fn = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # -------- 3) MCC curve -------- #
    try:
        plt.figure()
        plt.plot(epochs, mcc_tr, label="Train")
        plt.plot(epochs, mcc_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("SPR_BENCH Training vs Validation MCC")
        plt.legend()
        plt.tight_layout()
        fn = os.path.join(working_dir, "SPR_BENCH_MCC_curve.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # -------- 4) Final test metrics bar -------- #
    try:
        plt.figure()
        keys = ["loss", "acc", "MCC", "RMA"]
        vals = [test_metrics.get(k, np.nan) for k in keys]
        plt.bar(keys, vals)
        plt.title("SPR_BENCH Final Test Metrics")
        for i, v in enumerate(vals):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        plt.tight_layout()
        fn = os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
        plt.close()

    print("Plots saved to:", working_dir)
    print("Test metrics:", test_metrics)
