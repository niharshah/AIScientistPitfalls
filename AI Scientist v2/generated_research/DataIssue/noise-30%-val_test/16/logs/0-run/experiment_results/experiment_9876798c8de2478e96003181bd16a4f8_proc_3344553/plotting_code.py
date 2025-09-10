import matplotlib.pyplot as plt
import numpy as np
import os

# prepare paths and load data -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch nested dicts ----------------------------------------
def g(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


# iterate through stored runs -------------------------------------------------
for run_name, run_dict in experiment_data.items():
    for ds_name, ds_dict in run_dict.items():
        epochs = np.array(g(ds_dict, "epochs", default=[]))
        if epochs is None or len(epochs) == 0:
            continue

        # ----- 1. LOSS CURVE -------------------------------------------------
        try:
            plt.figure()
            plt.plot(epochs, ds_dict["losses"]["train"], label="train")
            plt.plot(epochs, ds_dict["losses"]["val"], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} Loss Curve")
            plt.legend()
            fname = f"{ds_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve: {e}")
            plt.close()

        # common routine to plot metrics -------------------------------------
        metric_names = ["acc", "MCC", "RMA"]
        for metric in metric_names:
            try:
                tr = [m[metric] for m in ds_dict["metrics"]["train"]]
                va = [m[metric] for m in ds_dict["metrics"]["val"]]
                plt.figure()
                plt.plot(epochs, tr, label="train")
                plt.plot(epochs, va, label="val")
                plt.xlabel("Epoch")
                plt.ylabel(metric)
                plt.title(f"{ds_name} {metric} Curve")
                plt.legend()
                fname = f"{ds_name}_{metric}_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating {metric} curve: {e}")
                plt.close()

        # ----- 5. CONFUSION MATRIX (optional) -------------------------------
        try:
            preds = np.array(ds_dict.get("predictions", []))
            gts = np.array(ds_dict.get("ground_truth", []))
            if preds.size and gts.size:
                cm = np.zeros((2, 2), dtype=int)
                for p, g in zip(preds, gts):
                    cm[int(g), int(p)] += 1
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        plt.text(
                            j, i, str(cm[i, j]), ha="center", va="center", color="black"
                        )
                plt.xticks([0, 1], ["Pred 0", "Pred 1"])
                plt.yticks([0, 1], ["True 0", "True 1"])
                plt.title(f"{ds_name} Confusion Matrix (Test)")
                plt.savefig(
                    os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
                )
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

        # print final test metrics -------------------------------------------
        test_met = ds_dict.get("test_metrics", {})
        if test_met:
            print(f"\n{ds_name} TEST METRICS:")
            for k, v in test_met.items():
                print(f"  {k}: {v:.4f}")
