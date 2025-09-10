import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------------- #
# Load experiment data
# ------------------------------------------------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to fetch safest value
def _get(lst, key, default=np.nan):
    return [d.get(key, default) for d in lst]


# ------------------------------------------------------------------------- #
# Iterate over datasets contained in bookkeeping dict
# ------------------------------------------------------------------------- #
for dset_name, log in experiment_data.items():
    # ----------------------- Plot 1: loss curves -------------------------- #
    try:
        plt.figure()
        epochs = range(1, len(log["losses"]["train"]) + 1)
        plt.plot(epochs, log["losses"]["train"], label="Train")
        plt.plot(epochs, log["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name} – Training vs Validation Loss")
        plt.legend()
        fname = f"{dset_name}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset_name}: {e}")
        plt.close()

    # --------------------- Plot 2: accuracy curves ------------------------ #
    try:
        plt.figure()
        train_acc = _get(log["metrics"]["train"], "acc")
        val_entries = log["metrics"]["val"]
        val_acc = _get(val_entries, "acc")
        val_cwa = _get(val_entries, "CWA")
        val_swa = _get(val_entries, "SWA")
        val_cswa = _get(val_entries, "CSWA")

        plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Val Acc")
        plt.plot(epochs, val_cwa, label="Val CWA")
        plt.plot(epochs, val_swa, label="Val SWA")
        plt.plot(epochs, val_cswa, label="Val CSWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dset_name} – Accuracy Metrics over Epochs")
        plt.legend()
        fname = f"{dset_name}_accuracy_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dset_name}: {e}")
        plt.close()

    # --------------- Plot 3: bar chart of final test metrics ------------- #
    try:
        plt.figure()
        test_metrics = val_entries[-1] if val_entries else {}
        labels = ["Acc", "CWA", "SWA", "CSWA"]
        values = [
            test_metrics.get("acc", 0),
            test_metrics.get("CWA", 0),
            test_metrics.get("SWA", 0),
            test_metrics.get("CSWA", 0),
        ]
        plt.bar(
            labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red"]
        )
        plt.ylim(0, 1)
        plt.title(f"{dset_name} – Final Validation Metrics")
        fname = f"{dset_name}_final_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric bar plot for {dset_name}: {e}")
        plt.close()

    # ---------------------------- Print summary -------------------------- #
    best_epoch = log.get("best_epoch", "n/a")
    print(f"{dset_name} best epoch: {best_epoch}")
    if log["metrics"]["val"]:
        best_vals = log["metrics"]["val"][
            best_epoch - 1 if isinstance(best_epoch, int) else -1
        ]
        print(
            f"  Validation Acc: {best_vals.get('acc', np.nan):.3f}  "
            f"CWA: {best_vals.get('CWA', np.nan):.3f}  "
            f"SWA: {best_vals.get('SWA', np.nan):.3f}  "
            f"CSWA: {best_vals.get('CSWA', np.nan):.3f}"
        )
