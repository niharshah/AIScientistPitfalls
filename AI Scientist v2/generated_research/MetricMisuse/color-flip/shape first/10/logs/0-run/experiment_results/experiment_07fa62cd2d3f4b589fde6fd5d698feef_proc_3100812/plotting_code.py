import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

if "SPR_BENCH" not in experiment_data:
    print("SPR_BENCH entry not found in experiment_data.npy")
    exit()

bench = experiment_data["SPR_BENCH"]

pre_loss = bench["losses"].get("pretrain", [])
train_loss = bench["losses"].get("train", [])
val_loss = bench["losses"].get("val", [])
swa = bench["metrics"].get("SWA", [])
cwa = bench["metrics"].get("CWA", [])
scwa = bench["metrics"].get("SCWA", [])
gt = np.array(bench.get("ground_truth", []))
pr = np.array(bench.get("predictions", []))

# helper for epoch index
epochs_pre = np.arange(1, len(pre_loss) + 1)
epochs_sup = np.arange(1, len(train_loss) + 1)

# 1) Pre-training loss ---------------------------------------------------------
try:
    if len(pre_loss):
        plt.figure()
        plt.plot(epochs_pre, pre_loss, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Contrastive Loss")
        plt.title("SPR_BENCH: Pre-training Contrastive Loss\n(Lower is Better)")
        fname = os.path.join(working_dir, "spr_bench_pretrain_loss_curve.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating pretrain loss plot: {e}")
    plt.close()

# 2) Training vs Validation loss ----------------------------------------------
try:
    if len(train_loss) and len(val_loss):
        plt.figure()
        plt.plot(epochs_sup, train_loss, label="Train", marker="o")
        plt.plot(epochs_sup, val_loss, label="Validation", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Supervised Loss Curves\n(Solid: Train, Dashed: Val)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_train_val_loss_curve.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating train/val loss plot: {e}")
    plt.close()

# 3) Weighted accuracy metrics -------------------------------------------------
try:
    if len(swa) and len(cwa) and len(scwa):
        plt.figure()
        plt.plot(epochs_sup, swa, label="SWA", marker="o")
        plt.plot(epochs_sup, cwa, label="CWA", marker="^")
        plt.plot(epochs_sup, scwa, label="SCWA", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Weighted Accuracy Metrics\n(Higher is Better)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_weighted_accuracy_metrics.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# 4) Per-class accuracy --------------------------------------------------------
try:
    if gt.size and pr.size:
        classes = sorted(set(gt))
        acc_per_class = [
            (pr[gt == cls] == cls).mean() if (gt == cls).sum() else 0 for cls in classes
        ]
        x = np.arange(len(classes))
        plt.figure()
        plt.bar(x, acc_per_class, tick_label=classes)
        plt.ylim(0, 1)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Per-Class Accuracy\n(Last Epoch Predictions)")
        fname = os.path.join(working_dir, "spr_bench_per_class_accuracy.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
except Exception as e:
    print(f"Error creating per-class accuracy plot: {e}")
    plt.close()

# ------------ Print final metrics --------------------------------------------
if gt.size and pr.size:
    overall_acc = (gt == pr).mean()
    print(f"Final Overall Accuracy: {overall_acc:.3f}")
if swa:
    print(
        f"Final SWA={swa[-1]:.3f} | CWA={cwa[-1]:.3f} | SCWA={scwa[-1]:.3f}"
        if cwa and scwa
        else ""
    )
