import matplotlib.pyplot as plt
import numpy as np
import os

# paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to optionally down-sample epochs to ≤ 100 points for clarity
def trim(xs, ys, max_pts=100):
    if len(xs) <= max_pts:
        return xs, ys
    step = max(1, len(xs) // max_pts)
    return xs[::step], ys[::step]


for ds_name, ds_data in experiment_data.get("epochs", {}).items():
    losses_tr = np.array(ds_data["losses"]["train"])
    losses_val = np.array(ds_data["losses"]["val"])
    acc_tr = np.array([m["acc"] for m in ds_data["metrics"]["train"]])
    acc_val = np.array([m["acc"] for m in ds_data["metrics"]["val"]])
    cwa = np.array([m["cwa"] for m in ds_data["metrics"]["val"]])
    swa = np.array([m["swa"] for m in ds_data["metrics"]["val"]])
    caa = np.array([m["caa"] for m in ds_data["metrics"]["val"]])
    epochs = np.arange(1, len(losses_tr) + 1)

    # 1) loss curves
    try:
        plt.figure()
        xs, tr = trim(epochs, losses_tr)
        _, va = trim(epochs, losses_val)
        plt.plot(xs, tr, label="Train")
        plt.plot(xs, va, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Dataset Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) accuracy curves
    try:
        plt.figure()
        xs, tr = trim(epochs, acc_tr)
        _, va = trim(epochs, acc_val)
        plt.plot(xs, tr, label="Train")
        plt.plot(xs, va, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Dataset Training vs Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # 3) weighted accuracy variants
    try:
        plt.figure()
        xs, c = trim(epochs, cwa)
        _, s = trim(epochs, swa)
        _, k = trim(epochs, caa)
        plt.plot(xs, c, label="CWA")
        plt.plot(xs, s, label="SWA")
        plt.plot(xs, k, label="CAA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(f"{ds_name} Dataset Validation Weighted Accuracies")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_weighted_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating weighted accuracy plot for {ds_name}: {e}")
        plt.close()

    # print evaluation summary
    best_val_acc = acc_val.max() if acc_val.size else float("nan")
    print(
        f"{ds_name}: best_val_acc={best_val_acc:.3f}, "
        f"final_val_acc={acc_val[-1]:.3f}, final_train_acc={acc_tr[-1]:.3f}"
    )
