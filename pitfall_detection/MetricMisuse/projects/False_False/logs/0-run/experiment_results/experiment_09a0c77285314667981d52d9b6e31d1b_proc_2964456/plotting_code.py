import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    edata = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    lr_dict = edata["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    lr_dict = {}

# Collect basic info
lr_keys = sorted(lr_dict.keys(), key=lambda x: float(x.split("_")[1]))
epochs = max(len(lr_dict[k]["losses"]["train"]) for k in lr_keys) if lr_keys else 0

# --------- Plot 1: Loss curves ----------
try:
    plt.figure()
    for k in lr_keys:
        lr_val = k.split("_")[1]
        train_loss = lr_dict[k]["losses"]["train"]
        val_loss = lr_dict[k]["losses"]["val"]
        ep = np.arange(1, len(train_loss) + 1)
        plt.plot(ep, train_loss, label=f"Train (lr={lr_val})", linestyle="--")
        plt.plot(ep, val_loss, label=f"Val   (lr={lr_val})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs. Validation Loss")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------- Plot 2: Macro-F1 curves ----------
try:
    plt.figure()
    for k in lr_keys:
        lr_val = k.split("_")[1]
        tr_f1 = lr_dict[k]["metrics"]["train_macroF1"]
        val_f1 = lr_dict[k]["metrics"]["val_macroF1"]
        ep = np.arange(1, len(tr_f1) + 1)
        plt.plot(ep, tr_f1, label=f"Train (lr={lr_val})", linestyle="--")
        plt.plot(ep, val_f1, label=f"Val   (lr={lr_val})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs. Validation Macro-F1")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating macroF1 plot: {e}")
    plt.close()

# --------- Plot 3: SWA & CWA bar chart ----------
try:
    ind = np.arange(len(lr_keys))
    width = 0.35
    swa_vals = [lr_dict[k]["shape_weighted_acc"] for k in lr_keys]
    cwa_vals = [lr_dict[k]["color_weighted_acc"] for k in lr_keys]

    plt.figure()
    plt.bar(ind - width / 2, swa_vals, width, label="Shape-Weighted Acc")
    plt.bar(ind + width / 2, cwa_vals, width, label="Color-Weighted Acc")
    plt.xticks(ind, [k.split("_")[1] for k in lr_keys])
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Weighted Accuracies at Final Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_weighted_accuracies.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close()

# --------- Print summary metrics ----------
for k in lr_keys:
    lr_val = k.split("_")[1]
    best_val_f1 = max(lr_dict[k]["metrics"]["val_macroF1"])
    swa = lr_dict[k]["shape_weighted_acc"]
    cwa = lr_dict[k]["color_weighted_acc"]
    print(
        f"lr={lr_val}: best Val Macro-F1={best_val_f1:.4f} | SWA={swa:.4f} | CWA={cwa:.4f}"
    )
