import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def get_first(d):
    # recursively get first (key, value) pair
    if isinstance(d, dict):
        k = next(iter(d))
        return get_first(d[k])
    return d


# ---------- iterate models/datasets ----------
for model_name, model_dict in experiment_data.items():
    for dset_name, ed in model_dict.items():
        epochs = ed.get("epochs", [])
        losses = ed.get("losses", {})
        metrics = ed.get("metrics", {})
        # ---------- plots ----------
        # 1) Loss curves
        try:
            plt.figure()
            if losses.get("train"):
                plt.plot(epochs, losses["train"], label="Train")
            if losses.get("val"):
                plt.plot(epochs, losses["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_name} Loss Curves\nLeft: Train, Right: Validation")
            plt.legend()
            fname = f"{dset_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # 2-4) Metric curves
        for mk in ["CWA", "SWA", "CplxWA"]:
            try:
                tr = metrics.get("train", {}).get(mk, [])
                vl = metrics.get("val", {}).get(mk, [])
                if tr or vl:
                    plt.figure()
                    if tr:
                        plt.plot(epochs, tr, label="Train")
                    if vl:
                        plt.plot(epochs, vl, label="Validation")
                    plt.xlabel("Epoch")
                    plt.ylabel(mk)
                    plt.title(
                        f"{dset_name} {mk} Over Epochs\nLeft: Train, Right: Validation"
                    )
                    plt.legend()
                    fname = f"{dset_name}_{mk}_curves.png"
                    plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating {mk} plot: {e}")
                plt.close()

        # ---------- print final test metrics ----------
        test_m = metrics.get("test", {})
        if test_m:
            print(f"{model_name}-{dset_name} Test metrics:", test_m)
