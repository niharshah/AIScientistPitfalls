import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# loop through every dataset stored in the npy file
for dname, dct in experiment_data.items():
    # ---------------- Loss curves ----------------
    try:
        if "losses" in dct and "train" in dct["losses"]:
            plt.figure()
            plt.plot(dct["losses"]["train"], label="train")
            plt.plot(dct["losses"]["val"], label="val", linestyle="--")
            plt.title(f"{dname} Loss Curves\nTrain vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # ---------------- Accuracy curves -------------
    try:
        if "metrics" in dct and "train" in dct["metrics"]:
            plt.figure()
            plt.plot(dct["metrics"]["train"], label="train")
            plt.plot(dct["metrics"]["val"], label="val", linestyle="--")
            plt.title(f"{dname} Accuracy Curves\nTrain vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
        plt.close()

    # ---------------- SWA curves ------------------
    try:
        if "swa" in dct and "train" in dct["swa"]:
            plt.figure()
            plt.plot(dct["swa"]["train"], label="train")
            plt.plot(dct["swa"]["val"], label="val", linestyle="--")
            plt.title(f"{dname} Shape-Weighted Accuracy (SWA)")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_swa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {dname}: {e}")
        plt.close()

    # -------- Final test metrics grouped bar -------
    try:
        if "test_metrics" in dct:
            tm = dct["test_metrics"]
            keys, vals = zip(*tm.items())
            plt.figure()
            plt.bar(np.arange(len(vals)), vals, color="skyblue")
            plt.xticks(np.arange(len(vals)), keys)
            plt.title(f"{dname} Final Test Metrics")
            plt.ylabel("Score")
            plt.savefig(os.path.join(working_dir, f"{dname}_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar for {dname}: {e}")
        plt.close()

    # ---- optional scatter SWA vs Accuracy ---------
    try:
        if {"metrics", "swa"} <= dct.keys():
            acc = dct["metrics"]["val"]
            swa = dct["swa"]["val"]
            if len(acc) == len(swa):
                plt.figure()
                plt.scatter(acc, swa, c=range(len(acc)), cmap="viridis")
                plt.colorbar(label="Epoch")
                plt.title(f"{dname} Val Accuracy vs SWA")
                plt.xlabel("Accuracy")
                plt.ylabel("SWA")
                plt.savefig(os.path.join(working_dir, f"{dname}_acc_vs_swa.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating acc-vs-swa scatter for {dname}: {e}")
        plt.close()

    # ------------- quick console print -------------
    if "test_metrics" in dct:
        print(f"{dname} test metrics:", dct["test_metrics"])
