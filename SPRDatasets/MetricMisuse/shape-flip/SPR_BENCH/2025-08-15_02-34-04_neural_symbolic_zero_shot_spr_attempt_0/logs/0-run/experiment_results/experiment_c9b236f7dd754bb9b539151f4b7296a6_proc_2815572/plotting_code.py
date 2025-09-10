import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    sweep = experiment_data["weight_decay_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    sweep = {}


# ------------------------------------------------------------
# Helper to pick at most 5 evenly spaced keys
def select_keys(keys, k=5):
    if len(keys) <= k:
        return keys
    idx = np.linspace(0, len(keys) - 1, k, dtype=int)
    return [keys[i] for i in idx]


# ------------------------------------------------------------
# PLOT 1: final val HWA vs weight_decay
try:
    wds = sorted(sweep.keys(), key=lambda s: float(s.split("=")[1]))
    wd_vals = [float(k.split("=")[1]) for k in wds]
    val_hwa = [sweep[k]["metrics"]["val"][-1][2] for k in wds]

    plt.figure()
    plt.plot(wd_vals, val_hwa, marker="o")
    plt.xscale("log")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Validation HWA")
    plt.title(
        "SPR_BENCH: Final Validation HWA vs Weight Decay\nLeft: Weight Decay (log), Right: HWA"
    )
    fname = os.path.join(working_dir, "spr_bench_val_hwa_vs_wd.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# ------------------------------------------------------------
# PLOT 2: validation & training HWA curves for up to 5 wds
try:
    sel_keys = select_keys(wds, 5)
    plt.figure()
    for k in sel_keys:
        epochs = range(1, len(sweep[k]["metrics"]["val"]) + 1)
        val_curve = [m[2] for m in sweep[k]["metrics"]["val"]]
        tr_curve = [m[2] for m in sweep[k]["metrics"]["train"]]
        wd_val = float(k.split("=")[1])
        plt.plot(epochs, val_curve, label=f"val wd={wd_val}")
        plt.plot(epochs, tr_curve, linestyle="--", label=f"train wd={wd_val}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title(
        "SPR_BENCH: HWA Curves Across Epochs\nLeft: Training (--), Right: Validation"
    )
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "spr_bench_hwa_curves_selected_wd.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# ------------------------------------------------------------
# PLOT 3: loss curves for best wd
try:
    best_k = max(wds, key=lambda k: sweep[k]["metrics"]["val"][-1][2])
    losses_tr = sweep[best_k]["losses"]["train"]
    losses_val = sweep[best_k]["losses"]["val"]
    epochs = range(1, len(losses_tr) + 1)

    plt.figure()
    plt.plot(epochs, losses_tr, label="train loss")
    plt.plot(epochs, losses_val, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        f"SPR_BENCH: Loss Curves for Best Weight Decay ({best_k})\nLeft: Training, Right: Validation"
    )
    plt.legend()
    fname = os.path.join(
        working_dir, f'spr_bench_loss_curves_{best_k.replace("=","")}.png'
    )
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()
