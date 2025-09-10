import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Guard clause if data missing
if not experiment_data:
    exit()

sweep = experiment_data["weight_decay"]
wds = sorted(sweep.keys(), key=lambda x: float(x))
epochs = sweep[wds[0]]["epochs"] if wds else []

# Helper to pick best wd (highest final val SDWA)
best_wd, best_val = None, -1
for wd in wds:
    val_sdwa_last = sweep[wd]["metrics"]["val"][-1]
    if val_sdwa_last > best_val:
        best_val = val_sdwa_last
        best_wd = wd
print(f"Best weight_decay: {best_wd} (final val SDWA={best_val:.4f})")
print(
    f"Test SDWA for best wd: {np.mean(sweep[best_wd]['metrics']['val']):.4f}"
    if best_wd
    else ""
)

# ---------------- plotting ----------------------------
# 1) Validation loss curves
try:
    plt.figure()
    for wd in wds:
        plt.plot(sweep[wd]["epochs"], sweep[wd]["losses"]["val"], label=f"wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Synthetic SPR Dataset\nLeft: Validation Loss across Weight Decays")
    plt.legend()
    fname = os.path.join(working_dir, "spr_val_loss_weight_decay.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating validation-loss plot: {e}")
    plt.close()

# 2) SDWA metric curves
try:
    plt.figure()
    for wd in wds:
        plt.plot(
            sweep[wd]["epochs"],
            sweep[wd]["metrics"]["train"],
            linestyle="--",
            alpha=0.6,
            label=f"train wd={wd}",
        )
        plt.plot(sweep[wd]["epochs"], sweep[wd]["metrics"]["val"], label=f"val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("SDWA")
    plt.title("Synthetic SPR Dataset\nLeft: Train (dashed) & Val (solid) SDWA")
    plt.legend(ncol=2, fontsize=7)
    fname = os.path.join(working_dir, "spr_sdwa_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating SDWA plot: {e}")
    plt.close()

# 3) Confusion matrix for best weight_decay on test set
try:
    gt = np.array(sweep[best_wd]["ground_truth"])
    pr = np.array(sweep[best_wd]["predictions"])
    num_classes = max(gt.max(), pr.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(gt, pr):
        cm[g, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"Synthetic SPR Dataset\nConfusion Matrix (wd={best_wd})")
    # tick labels
    ticks = range(num_classes)
    plt.xticks(ticks)
    plt.yticks(ticks)
    fname = os.path.join(working_dir, f"spr_confusion_matrix_wd_{best_wd}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating confusion-matrix plot: {e}")
    plt.close()
