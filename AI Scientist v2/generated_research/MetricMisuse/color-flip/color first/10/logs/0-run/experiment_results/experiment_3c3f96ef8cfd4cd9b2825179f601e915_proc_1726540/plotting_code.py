import matplotlib.pyplot as plt
import numpy as np
import os
import re

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def _num(wd_key):
    try:
        return float(wd_key.split("_")[1])
    except Exception:
        # fallback: extract with regex
        return float(re.findall(r"[-+e0-9.]+", wd_key)[0])


try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["weight_decay"]
    wd_keys = sorted(runs.keys(), key=_num)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs, wd_keys = {}, []

# Figure 1: loss curves --------------------------------------------------------
try:
    plt.figure()
    for k in wd_keys:
        epochs = range(1, len(runs[k]["losses"]["train"]) + 1)
        plt.plot(epochs, runs[k]["losses"]["train"], label=f"train wd={_num(k):g}")
        plt.plot(
            epochs,
            runs[k]["losses"]["val"],
            linestyle="--",
            label=f"val wd={_num(k):g}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("weight_decay: Train vs Val Loss")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "weight_decay_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Figure 2: CompWA curves ------------------------------------------------------
try:
    plt.figure()
    for k in wd_keys:
        epochs = range(1, len(runs[k]["metrics"]["train_CompWA"]) + 1)
        plt.plot(
            epochs, runs[k]["metrics"]["train_CompWA"], label=f"train wd={_num(k):g}"
        )
        plt.plot(
            epochs,
            runs[k]["metrics"]["val_CompWA"],
            linestyle="--",
            label=f"val wd={_num(k):g}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.title("weight_decay: Train vs Val CompWA")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "weight_decay_compwa_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating CompWA curves: {e}")
    plt.close()

# Figure 3: final val CompWA bar chart ----------------------------------------
try:
    plt.figure()
    vals = [runs[k]["metrics"]["val_CompWA"][-1] for k in wd_keys]
    plt.bar(range(len(wd_keys)), vals, tick_label=[f"{_num(k):g}" for k in wd_keys])
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Val CompWA")
    plt.title("weight_decay: Final Validation CompWA per WD")
    fname = os.path.join(working_dir, "weight_decay_final_compwa.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating final CompWA bar chart: {e}")
    plt.close()

# Figure 4: CWA & SWA comparison ----------------------------------------------
try:
    plt.figure()
    x = np.arange(len(wd_keys))
    width = 0.35
    cwa = [runs[k]["CWA"] for k in wd_keys]
    swa = [runs[k]["SWA"] for k in wd_keys]
    plt.bar(x - width / 2, cwa, width=width, label="CWA")
    plt.bar(x + width / 2, swa, width=width, label="SWA")
    plt.xticks(x, [f"{_num(k):g}" for k in wd_keys])
    plt.xlabel("Weight Decay")
    plt.ylabel("Weighted Accuracy")
    plt.title("weight_decay: CWA vs SWA")
    plt.legend()
    fname = os.path.join(working_dir, "weight_decay_cwa_swa.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating CWA/SWA comparison: {e}")
    plt.close()
