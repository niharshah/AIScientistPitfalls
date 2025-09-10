import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------#
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper: gather per-budget info
final_test_acc = {}
best_tag, best_val_loss = None, float("inf")

for tag, rec in experiment_data.get("epochs", {}).items():
    val_losses = rec["losses"]["val"]
    if val_losses and min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)
        best_tag = tag
    # final test accuracy is last element of metrics['val']? we stored separately in print, but gather here from val
    # safer: compute from predictions vs ground_truth
    preds, gts = np.array(rec["predictions"]), np.array(rec["ground_truth"])
    if len(preds) == len(gts) and len(preds):
        final_test_acc[tag] = (preds == gts).mean()

print("Final test accuracies:")
for t, acc in final_test_acc.items():
    print(f"  {t}: {acc*100:.2f}%")
print(f"Best tag by val loss: {best_tag} (min val loss {best_val_loss:.4f})")

# ------------------------------------------------------------------#
# 1) Loss-vs-Epoch plot
try:
    plt.figure()
    for tag, rec in experiment_data.get("epochs", {}).items():
        plt.plot(rec["losses"]["train"], label=f"{tag}-train")
        plt.plot(rec["losses"]["val"], linestyle="--", label=f"{tag}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves across Epoch Budgets\nDataset: SPR_BENCH")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------#
# 2) Accuracy-vs-Epoch plot
try:
    plt.figure()
    for tag, rec in experiment_data.get("epochs", {}).items():
        plt.plot(rec["metrics"]["train"], label=f"{tag}-train")
        plt.plot(rec["metrics"]["val"], linestyle="--", label=f"{tag}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves across Epoch Budgets\nDataset: SPR_BENCH")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------#
# 3) Confusion matrix for best model
try:
    if best_tag and best_tag in experiment_data["epochs"]:
        rec = experiment_data["epochs"][best_tag]
        preds, gts = np.array(rec["predictions"]), np.array(rec["ground_truth"])
        num_labels = int(max(gts.max(), preds.max())) + 1 if len(gts) else 0
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for i in range(num_labels):
            for j in range(num_labels):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"Confusion Matrix (Best: {best_tag})\nDataset: SPR_BENCH")
        fname = os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_{best_tag}.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
