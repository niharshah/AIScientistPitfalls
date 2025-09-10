import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ---------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("d_model_tuning", {}).get("SPR_BENCH", {})
if not spr_data:
    print("No SPR_BENCH data available.")
    exit()

tags = sorted(spr_data.keys(), key=lambda x: int(x.split("_")[-1]))  # e.g. d_model_32
epochs = len(next(iter(spr_data.values()))["losses"]["train"])

# ---------------- figure 1: loss curves ---------------- #
try:
    plt.figure(figsize=(6, 4))
    for tag in tags:
        ep = np.arange(1, epochs + 1)
        plt.plot(
            ep, spr_data[tag]["losses"]["train"], label=f"{tag}-train", linestyle="-"
        )
        plt.plot(ep, spr_data[tag]["losses"]["val"], label=f"{tag}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Val Loss (d_model sweep)")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_d_model_sweep.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------- figure 2: accuracy curves ---------------- #
try:
    plt.figure(figsize=(6, 4))
    for tag in tags:
        ep = np.arange(1, epochs + 1)
        plt.plot(
            ep, spr_data[tag]["metrics"]["train"], label=f"{tag}-train", linestyle="-"
        )
        plt.plot(
            ep, spr_data[tag]["metrics"]["val"], label=f"{tag}-val", linestyle="--"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Train vs Val Accuracy (d_model sweep)")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves_d_model_sweep.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ---------------- figure 3: final test accuracy bar chart ---------------- #
try:
    test_accs = [spr_data[tag]["metrics"]["val"][-1] for tag in tags]  # val final epoch
    plt.figure(figsize=(5, 3.5))
    plt.bar(tags, test_accs, color="skyblue")
    plt.ylabel("Validation Accuracy at Last Epoch")
    plt.title("SPR_BENCH: Final Validation Accuracy by d_model")
    plt.xticks(rotation=45, ha="right")
    fname = os.path.join(working_dir, "SPR_BENCH_final_val_accuracy_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ---------------- confusion matrix for best model ---------------- #
try:
    # Determine best d_model by best validation accuracy last epoch
    best_idx = int(np.argmax(test_accs))
    best_tag = tags[best_idx]
    preds = np.array(spr_data[best_tag]["predictions"])
    gts = np.array(spr_data[best_tag]["ground_truth"])
    num_labels = len(np.unique(gts))
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for gt, pr in zip(gts, preds):
        cm[gt, pr] += 1

    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH Confusion Matrix (best {best_tag})")
    for i in range(num_labels):
        for j in range(num_labels):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
    fname = os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_{best_tag}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- print evaluation metric ---------------- #
best_val_acc = test_accs[best_idx]
print(
    f"Best d_model: {best_tag}  |  Final validation accuracy: {best_val_acc*100:.2f}%"
)
