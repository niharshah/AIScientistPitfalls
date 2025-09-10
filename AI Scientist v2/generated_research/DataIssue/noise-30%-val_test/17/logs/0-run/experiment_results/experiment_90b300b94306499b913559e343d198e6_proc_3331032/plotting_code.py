import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
data = experiment_data.get(ds_name, {})


# helper to get arrays safely
def get_arr(path, default=None):
    cur = data
    for key in path:
        cur = cur.get(key, {})
    return np.array(cur if isinstance(cur, (list, np.ndarray)) else default)


epochs = np.arange(1, len(get_arr(["losses", "train"])) + 1)

# 1) loss curve
try:
    plt.figure()
    plt.plot(epochs, get_arr(["losses", "train"]), label="Train")
    plt.plot(epochs, get_arr(["losses", "val"]), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"{ds_name} Loss Curve\nTrain vs. Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name.lower()}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) MCC curve
try:
    plt.figure()
    plt.plot(epochs, get_arr(["metrics", "train_mcc"]), label="Train")
    plt.plot(epochs, get_arr(["metrics", "val_mcc"]), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews CorrCoef")
    plt.title(f"{ds_name} MCC Curve\nTrain vs. Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name.lower()}_mcc_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating MCC curve: {e}")
    plt.close()

# 3) Macro-F1 curve
try:
    plt.figure()
    plt.plot(epochs, get_arr(["metrics", "train_f1"]), label="Train")
    plt.plot(epochs, get_arr(["metrics", "val_f1"]), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"{ds_name} Macro-F1 Curve\nTrain vs. Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name.lower()}_f1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 4) summary test bar plot (best run)
try:
    test_info = data.get("test", {})
    metrics = ["mcc", "macro_f1"]
    values = [test_info.get(m, np.nan) for m in metrics]
    plt.figure()
    plt.bar(metrics, values, color=["skyblue", "salmon"])
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.title(f"{ds_name} Best Test Performance")
    fname = os.path.join(working_dir, f"{ds_name.lower()}_test_summary.png")
    plt.savefig(fname)
    plt.close()
    print(
        f"Best Test MCC: {test_info.get('mcc', 'n/a'):.4f}, "
        f"Best Test macro-F1: {test_info.get('macro_f1', 'n/a'):.4f}"
    )
except Exception as e:
    print(f"Error creating test summary plot: {e}")
    plt.close()
