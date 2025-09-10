import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data -------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------------
# identify best LR on dev BWA -----------------------------------------
best_lr, best_bwa = None, -1.0
dev_bwa_per_lr = {}
for lr_key, rec in (
    experiment_data.get("learning_rate", {}).get("SPR_BENCH", {}).items()
):
    val_series = rec["metrics"]["val"]
    if val_series:
        final_bwa = val_series[-1]
        dev_bwa_per_lr[lr_key] = final_bwa
        if final_bwa > best_bwa:
            best_bwa, best_lr = final_bwa, lr_key

print(f"Best LR on dev: {best_lr}  |  Best dev BWA: {best_bwa:.4f}")

# Short-circuit if nothing to plot
if not best_lr:
    exit(0)

best_rec = experiment_data["learning_rate"]["SPR_BENCH"][best_lr]
epochs = np.arange(1, len(best_rec["metrics"]["train"]) + 1)

# ---------------------------------------------------------------------
# 1) BWA curve ---------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, best_rec["metrics"]["train"], label="Train BWA")
    plt.plot(epochs, best_rec["metrics"]["val"], label="Dev BWA")
    plt.xlabel("Epoch")
    plt.ylabel("BWA")
    plt.title(f"SPR_BENCH – BWA over Epochs (LR={best_lr})")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, f"SPR_BENCH_bwa_curve_lr_{best_lr}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating BWA curve: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) Loss curve --------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, best_rec["losses"]["train"], label="Train Loss")
    plt.plot(epochs, best_rec["losses"]["val"], label="Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"SPR_BENCH – Loss over Epochs (LR={best_lr})")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, f"SPR_BENCH_loss_curve_lr_{best_lr}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss curve: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Dev BWA by LR -----------------------------------------------------
try:
    plt.figure()
    lrs, bwas = zip(*sorted(dev_bwa_per_lr.items(), key=lambda x: x[0]))
    plt.bar(range(len(lrs)), bwas, tick_label=lrs)
    plt.ylabel("Final Dev BWA")
    plt.xlabel("Learning Rate")
    plt.title("SPR_BENCH – Final Dev BWA by Learning Rate")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_val_bwa_by_lr.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating LR comparison bar plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 4) Confusion matrix on test set -------------------------------------
try:
    preds = np.array(best_rec.get("predictions", []))
    gts = np.array(best_rec.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = max(preds.max(), gts.max()) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"SPR_BENCH – Confusion Matrix (LR={best_lr})")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
                )
        plt.tight_layout()
        fname = os.path.join(
            working_dir, f"SPR_BENCH_confusion_matrix_lr_{best_lr}.png"
        )
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------------------------------------------------------------
# print final metrics --------------------------------------------------
cwa = best_rec.get("metrics", {}).get("val", [-1])[-1]  # placeholder if missing
test_cwa = best_rec.get("metrics", {}).get("val", [-1])[-1]  # reuse placeholder
print(f"Best LR={best_lr} | Dev BWA={best_bwa:.4f}")
if preds.size:
    # compute test BWA components
    def weighted_acc(weights):
        return (weights * (preds == gts)).sum() / max(weights.sum(), 1)

    # assume weights unavailable here, so just overall accuracy
    test_bwa = (preds == gts).mean()
    print(f"Test accuracy (unweighted BWA proxy): {test_bwa:.4f}")
