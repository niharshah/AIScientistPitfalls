import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import islice

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for ds_name, ds_dict in experiment_data.items():
    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        plt.plot(ds_dict["losses"]["train"], label="train")
        plt.plot(ds_dict["losses"]["val"], label="val", linestyle="--")
        plt.title(f"{ds_name} Loss Curves\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"{ds_name}: error plotting losses -> {e}")
        plt.close()

    # ---------- Plot 2: Accuracy curves ----------
    try:
        if "metrics" in ds_dict:
            plt.figure()
            plt.plot(ds_dict["metrics"]["train"], label="train")
            plt.plot(ds_dict["metrics"]["val"], label="val", linestyle="--")
            plt.title(f"{ds_name} Accuracy Curves\nTrain vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curves.png"))
            plt.close()
    except Exception as e:
        print(f"{ds_name}: error plotting accuracy -> {e}")
        plt.close()

    # ---------- Plot 3: SWA curves ----------
    try:
        if "swa" in ds_dict:
            plt.figure()
            plt.plot(ds_dict["swa"]["train"], label="train")
            plt.plot(ds_dict["swa"]["val"], label="val", linestyle="--")
            plt.title(f"{ds_name} Shape-Weighted Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_swa_curves.png"))
            plt.close()
    except Exception as e:
        print(f"{ds_name}: error plotting SWA -> {e}")
        plt.close()

    # ---------- Plot 4: Final test metrics ----------
    try:
        if "test" in ds_dict:
            test_keys = [
                k for k in ("acc", "swa", "rgs", "cwa") if k in ds_dict["test"]
            ]
            scores = [ds_dict["test"][k] for k in test_keys]
            plt.figure()
            plt.bar(range(len(scores)), scores, tick_label=test_keys, color="skyblue")
            plt.title(f"{ds_name} Final Test Metrics")
            plt.ylabel("Score")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_test_metrics_bar.png"))
            plt.close()
    except Exception as e:
        print(f"{ds_name}: error plotting test metrics -> {e}")
        plt.close()

    # ---------- Plot 5: Confusion-style heat-maps over epochs ----------
    try:
        if "predictions" in ds_dict and ds_dict["predictions"]:
            ep_keys = sorted(
                ds_dict["predictions"].keys(),
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )
            # keep at most 5 evenly spaced epochs
            sel_keys = [
                ep_keys[int(i)]
                for i in np.linspace(0, len(ep_keys) - 1, num=min(5, len(ep_keys)))
            ]
            for ep in sel_keys:
                preds = np.array(ds_dict["predictions"][ep])
                # rudimentary class histogram instead of full confusion matrix (labels unavailable)
                bins = np.bincount(preds, minlength=max(preds) + 1)
                plt.figure()
                plt.bar(range(len(bins)), bins)
                plt.title(f"{ds_name} Prediction Distribution\n{ep}")
                plt.xlabel("Predicted Class ID")
                plt.ylabel("Count")
                fname = f"{ds_name}_pred_hist_{ep}.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
    except Exception as e:
        print(f"{ds_name}: error plotting prediction hists -> {e}")
        plt.close()
