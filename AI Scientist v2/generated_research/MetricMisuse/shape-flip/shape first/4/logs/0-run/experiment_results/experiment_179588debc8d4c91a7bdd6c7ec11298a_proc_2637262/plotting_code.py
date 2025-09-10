import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
    wd_vals = np.array(ed["values"])
    train_acc = np.array(ed["metrics"]["train_acc"])
    dev_acc = np.array(ed["metrics"]["dev_acc"])
    test_acc = np.array(ed["metrics"]["test_acc"])
    dev_rgs = np.array(ed["metrics"]["dev_rgs"])
    test_rgs = np.array(ed["metrics"]["test_rgs"])
    losses_tr = ed["losses"]["train"]
    losses_dev = ed["losses"]["dev"]
except Exception as e:
    print(f"Error loading or parsing experiment data: {e}")
    raise SystemExit()

# ------------------ Plot 1: Accuracy vs weight_decay ----------------------
try:
    plt.figure()
    plt.plot(wd_vals, train_acc, marker="o", label="Train")
    plt.plot(wd_vals, dev_acc, marker="s", label="Dev")
    plt.plot(wd_vals, test_acc, marker="^", label="Test")
    plt.xscale("log")
    plt.xlabel("Weight Decay")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy vs Weight Decay\nLeft: Train, Right: Dev/Test")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_acc_vs_weight_decay.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------ Plot 2: RGS vs weight_decay ---------------------------
try:
    plt.figure()
    plt.plot(wd_vals, dev_rgs, marker="o", label="Dev RGS")
    plt.plot(wd_vals, test_rgs, marker="s", label="Test RGS")
    plt.xscale("log")
    plt.xlabel("Weight Decay")
    plt.ylabel("RGS")
    plt.title(
        "SPR_BENCH Rule-Generalisation Score vs Weight Decay\nLeft: Dev, Right: Test"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_rgs_vs_weight_decay.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating RGS plot: {e}")
    plt.close()

# ------------------ Plot 3: Best run loss curves --------------------------
try:
    best_idx = int(np.argmax(dev_acc))
    best_wd = wd_vals[best_idx]
    tr_curve = losses_tr[best_idx]
    dv_curve = losses_dev[best_idx]
    plt.figure()
    plt.plot(tr_curve, marker="o", label="Train Loss")
    plt.plot(dv_curve, marker="s", label="Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        f"SPR_BENCH Loss Curves (Best Weight Decay={best_wd})\nLeft: Train, Right: Dev"
    )
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_BENCH_loss_curves_best_wd_{best_wd}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()
