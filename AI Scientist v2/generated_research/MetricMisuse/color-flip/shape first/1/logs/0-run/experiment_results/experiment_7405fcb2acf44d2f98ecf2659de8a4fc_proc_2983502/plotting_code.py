import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper: safe fetch
def get(dic, *keys, default=None):
    for k in keys:
        dic = dic.get(k, {})
    return dic if dic else default


# -------------------- iterate datasets ------------
for dname, dct in experiment_data.items():
    # -------- loss curves -----------
    try:
        plt.figure()
        # plot only if series exist
        for tag in ["pretrain", "train", "val"]:
            series = get(dct, "losses", tag, default=None)
            if series is not None and len(series):
                plt.plot(range(1, len(series) + 1), series, label=tag)
        plt.title(f"{dname} Loss Curves\nLeft: Pre-training vs Fine-tuning")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # -------- metric curves ----------
    try:
        plt.figure()
        plotted = False
        for metric in ["SWA", "CWA", "SCHM"]:
            series = get(dct, "metrics", metric, default=None)
            if series is not None and len(series):
                plt.plot(range(1, len(series) + 1), series, label=metric)
                plotted = True
        if plotted:
            plt.title(f"{dname} Validation Metrics\nLeft: SWA, CWA, Right: SCHM")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_metric_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {dname}: {e}")
        plt.close()

    # -------- print final metrics ----
    final_metrics = {
        m: get(dct, "metrics", m, default=[None])[-1]
        for m in ["SWA", "CWA", "SCHM"]
        if get(dct, "metrics", m)
    }
    print(f"Final metrics for {dname}: {final_metrics}")
