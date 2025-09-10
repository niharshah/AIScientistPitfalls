import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["weight_decay"]["SPR_BENCH"]
    wds = sorted(spr_data.keys(), key=lambda x: float(x))
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data, wds = {}, []


# ---------------- helper --------------------
def get_curve(wd_key, field, split):
    """Return y-values curve for given weight-decay key."""
    return spr_data[wd_key][field][split]


# -------- 1. Loss curves --------------------
try:
    plt.figure()
    for wd in wds:
        epochs = spr_data[wd]["epochs"]
        plt.plot(epochs, get_curve(wd, "losses", "train"), label=f"train wd={wd}")
        plt.plot(
            epochs, get_curve(wd, "losses", "val"), linestyle="--", label=f"val wd={wd}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------- 2. F1 curves ----------------------
try:
    plt.figure()
    for wd in wds:
        epochs = spr_data[wd]["epochs"]
        plt.plot(epochs, get_curve(wd, "metrics", "train"), label=f"train wd={wd}")
        plt.plot(
            epochs,
            get_curve(wd, "metrics", "val"),
            linestyle="--",
            label=f"val wd={wd}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation F1")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# -------- 3. Dev F1 vs WD -------------------
try:
    dev_f1 = [get_curve(wd, "metrics", "val")[-1] for wd in wds]
    plt.figure()
    plt.plot([float(w) for w in wds], dev_f1, marker="o")
    plt.xscale("log")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Dev Macro-F1")
    plt.title("SPR_BENCH: Dev F1 vs Weight Decay")
    fname = os.path.join(working_dir, "SPR_BENCH_devF1_vs_wd.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating Dev-F1 plot: {e}")
    plt.close()

# -------- 4. Test F1 vs WD ------------------
try:
    test_f1 = [spr_data[wd]["test_f1"] for wd in wds]
    plt.figure()
    plt.plot([float(w) for w in wds], test_f1, marker="s", color="green")
    plt.xscale("log")
    plt.xlabel("Weight Decay")
    plt.ylabel("Test Macro-F1")
    plt.title("SPR_BENCH: Test F1 vs Weight Decay")
    fname = os.path.join(working_dir, "SPR_BENCH_testF1_vs_wd.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Test F1 per weight decay:", dict(zip(wds, test_f1)))
except Exception as e:
    print(f"Error creating Test-F1 plot: {e}")
    plt.close()

# -------- 5. Confusion matrix ---------------
try:
    # pick best wd on dev
    best_idx = int(np.argmax(dev_f1))
    best_wd = wds[best_idx]
    preds = np.array(spr_data[best_wd]["predictions"])
    gts = np.array(spr_data[best_wd]["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for g, p in zip(gts, preds):
        cm[g, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.title(
        f"SPR_BENCH Confusion Matrix (Best WD={best_wd})\nLeft: Ground Truth, Right: Predicted"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.colorbar()
    fname = os.path.join(working_dir, f"SPR_BENCH_confusion_best_wd_{best_wd}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
