import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- set up paths ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    ed = experiment_data["SPR_BENCH"]
    tr_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    tr_acc = [m["acc"] for m in ed["metrics"]["train"]]
    val_acc = [m["acc"] for m in ed["metrics"]["val"]]
    tr_swa = [m["swa"] for m in ed["metrics"]["train"]]
    val_swa = [m["swa"] for m in ed["metrics"]["val"]]
    epochs = list(range(1, len(tr_loss) + 1))

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, tr_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.title("SPR_BENCH – Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_train_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3) Shape-Weighted Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, tr_swa, label="Train")
        plt.plot(epochs, val_swa, label="Validation")
        plt.title("SPR_BENCH – Training vs Validation Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_train_val_swa.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 4) Test metrics bar chart
    try:
        plt.figure()
        acc_test = ed["metrics"]["test"].get("acc", None)
        swa_test = ed["metrics"]["test"].get("swa", None)
        metrics, values = [], []
        if acc_test is not None:
            metrics.append("Accuracy")
            values.append(acc_test)
        if swa_test is not None:
            metrics.append("SWA")
            values.append(swa_test)
        plt.bar(range(len(values)), values, tick_label=metrics)
        plt.title("SPR_BENCH – Test Performance")
        plt.ylabel("Score")
        fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
        plt.close()
else:
    print("No valid experiment data found for plotting.")
