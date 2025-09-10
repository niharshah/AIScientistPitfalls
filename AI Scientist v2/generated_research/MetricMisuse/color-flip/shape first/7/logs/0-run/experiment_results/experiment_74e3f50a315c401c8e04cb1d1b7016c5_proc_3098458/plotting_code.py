import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

lr_dict = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})
if not lr_dict:
    print("No data found, aborting plots.")
    exit()


# helper to extract arrays -----------------------------------------------------
def tup2arr(tup_list, idx=1):
    """get x(epochs) and y(values[idx]) arrays from list[(ep,val,...)]"""
    if not tup_list:
        return [], []
    ep = [t[0] for t in tup_list]
    val = [t[idx] for t in tup_list]
    return ep, val


# ---------------------- Figure 1: loss curves ---------------------------------
try:
    plt.figure()
    for lr, rec in lr_dict.items():
        ep_tr, tr = tup2arr(rec["losses"]["train"])
        ep_val, val = tup2arr(rec["losses"]["val"])
        plt.plot(ep_tr, tr, "--", label=f"lr={lr} train")
        plt.plot(ep_val, val, "-", label=f"lr={lr} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("SPR_BENCH Loss Curves (Train vs. Validation)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ---------------------- Figure 2: HWA curves ----------------------------------
try:
    plt.figure()
    for lr, rec in lr_dict.items():
        ep, hwa = tup2arr(rec["metrics"]["val"], idx=3)
        plt.plot(ep, hwa, label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH Validation HWA Across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves plot: {e}")
    plt.close()

# ---------------------- Figure 3: Final HWA bar -------------------------------
try:
    plt.figure()
    lrs, final_hwa = [], []
    for lr, rec in lr_dict.items():
        if rec["metrics"]["val"]:
            final_hwa.append(rec["metrics"]["val"][-1][3])
            lrs.append(lr)
    plt.bar(range(len(lrs)), final_hwa, tick_label=lrs)
    plt.ylabel("Final HWA")
    plt.title("SPR_BENCH Final Epoch HWA by Learning Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# -------------- print final metrics to stdout ---------------------------------
for lr, hwa in zip(lrs, final_hwa):
    print(f"Learning rate {lr}: final HWA = {hwa:.4f}")
