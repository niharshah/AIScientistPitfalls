import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    lr_dict = experiment_data["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    lr_dict = {}

# --------- Plot 1: Loss curves ----------
try:
    plt.figure()
    for lr_key, logs in lr_dict.items():
        tr = np.array(logs["losses"]["train"])
        val = np.array(logs["losses"]["val"])
        if tr.size and val.size:
            plt.plot(tr[:, 0], tr[:, 1], label=f"{lr_key} train")
            plt.plot(val[:, 0], val[:, 1], "--", label=f"{lr_key} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training & Validation Loss\n(Hyper-param sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# --------- Plot 2: HWA curves ----------
try:
    plt.figure()
    for lr_key, logs in lr_dict.items():
        tr = np.array(logs["metrics"]["train"])
        val = np.array(logs["metrics"]["val"])
        if tr.size and val.size:
            plt.plot(tr[:, 0], tr[:, 1], label=f"{lr_key} train")
            plt.plot(val[:, 0], val[:, 1], "--", label=f"{lr_key} val")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Training & Validation HWA\n(Hyper-param sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# --------- Plot 3: Best validation HWA bar chart ----------
try:
    plt.figure()
    lrs, best_vals = [], []
    for lr_key, logs in lr_dict.items():
        val = np.array(logs["metrics"]["val"])
        if val.size:
            lrs.append(lr_key)
            best_vals.append(val[:, 1].max())
    if lrs:
        plt.bar(range(len(lrs)), best_vals, tick_label=lrs)
        plt.ylabel("Best Validation HWA")
        plt.title("SPR_BENCH: Best Validation HWA per Learning Rate")
        fname = os.path.join(working_dir, "SPR_BENCH_best_val_HWA.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best-val HWA bar chart: {e}")
    plt.close()
