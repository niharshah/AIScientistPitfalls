import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- paths -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- data loading ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----------- plotting per dataset ----------
for ds_name, ds in experiment_data.items():
    # ---------- 1. Loss curves ----------
    try:
        plt.figure()
        if "losses" in ds and ds["losses"]:  # assure key exists
            plt.plot(ds["losses"]["train"], label="train")
            plt.plot(ds["losses"]["val"], label="val", linestyle="--")
            plt.title(f"{ds_name} Loss Curves\nTrain vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy")
            plt.legend(fontsize=7)
            fname = f"{ds_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("losses not found")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {ds_name}: {e}")
        plt.close()

    # ---------- 2. Accuracy curves ----------
    try:
        plt.figure()
        if "metrics" in ds and ds["metrics"]:
            plt.plot(ds["metrics"]["train"], label="train")
            plt.plot(ds["metrics"]["val"], label="val", linestyle="--")
            plt.title(f"{ds_name} Accuracy Curves\nTrain vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(fontsize=7)
            fname = f"{ds_name}_accuracy_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("metrics not found")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves for {ds_name}: {e}")
        plt.close()

    # ---------- 3. Shape-Weighted Accuracy curves ----------
    try:
        plt.figure()
        if "swa" in ds and ds["swa"]:
            plt.plot(ds["swa"]["train"], label="train")
            plt.plot(ds["swa"]["val"], label="val", linestyle="--")
            plt.title(f"{ds_name} SWA Curves\nTrain vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.legend(fontsize=7)
            fname = f"{ds_name}_swa_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("swa not found")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curves for {ds_name}: {e}")
        plt.close()

    # ---------- 4. Final test metrics ----------
    try:
        plt.figure()
        if "test_metrics" in ds:
            tm = ds["test_metrics"]
            bars = ["loss", "acc", "swa"]
            values = [tm.get(k, np.nan) for k in bars]
            x = np.arange(len(bars))
            plt.bar(x, values, color="skyblue")
            plt.xticks(x, bars)
            plt.title(f"{ds_name} Final Test Metrics")
            plt.ylabel("Score")
            fname = f"{ds_name}_test_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
        else:
            raise ValueError("test_metrics not found")
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar plot for {ds_name}: {e}")
        plt.close()
