import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR"
ds = experiment_data.get(dataset, {})
loss_tr = ds.get("losses", {}).get("train", [])
loss_val = ds.get("losses", {}).get("val", [])
pc_tr = ds.get("metrics", {}).get("train", [])
pc_val = ds.get("metrics", {}).get("val", [])
preds = np.array(ds.get("predictions", []))
gts = np.array(ds.get("ground_truth", []))


# helper to pull y-values while keeping epoch order intact
def values(tuples):
    return [v for _, v in tuples]


# ---------------------------------------------------------------------
# 1) Loss curves -------------------------------------------------------
try:
    if loss_tr and loss_val:
        ep = np.arange(1, len(loss_tr) + 1)
        plt.figure(figsize=(7, 4))
        plt.plot(ep, values(loss_tr), "--", label="train")
        plt.plot(ep, values(loss_val), "-", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss - SPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) PCWA curves -------------------------------------------------------
try:
    if pc_tr and pc_val:
        ep = np.arange(1, len(pc_tr) + 1)
        plt.figure(figsize=(7, 4))
        plt.plot(ep, values(pc_tr), "--", label="train PCWA")
        plt.plot(ep, values(pc_val), "-", label="val PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.title("Training & Validation PCWA - SPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_pcwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PCWA curves plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Final PCWA bar chart ---------------------------------------------
try:
    finals = {}
    if pc_tr:
        finals["train"] = values(pc_tr)[-1]
    if pc_val:
        finals["val"] = values(pc_val)[-1]
    if preds.size and gts.size:
        # compute test PCWA exactly as training code did
        def count_color_variety(seq):
            return len(set(tok[1] for tok in seq.strip().split()))

        def count_shape_variety(seq):
            return len(set(tok[0] for tok in seq.strip().split()))

        seqs = ds.get("ground_truth_sequences", []) or []
        if not seqs:  # sequences may be missing; fall back to zeros
            seqs = [""] * len(preds)
        w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
        finals["test"] = (
            sum(wi if y == p else 0 for wi, y, p in zip(w, gts, preds)) / sum(w)
            if sum(w)
            else 0.0
        )
    if finals:
        plt.figure(figsize=(5, 4))
        plt.bar(finals.keys(), finals.values(), color="skyblue")
        plt.ylabel("Final PCWA")
        plt.title("Final PCWA Scores - SPR")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_final_pcwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final PCWA bar plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 4) Confusion matrix --------------------------------------------------
try:
    if preds.size and gts.size:
        classes = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for gt, pr in zip(gts, preds):
            i, j = np.where(classes == gt)[0][0], np.where(classes == pr)[0][0]
            cm[i, j] += 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix - SPR Test Set")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 5) Class distribution ------------------------------------------------
try:
    if preds.size and gts.size:
        labels, gt_counts = np.unique(gts, return_counts=True)
        _, pr_counts = np.unique(preds, return_counts=True)
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(6, 4))
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pr_counts, width, label="Predictions")
        plt.xticks(x, labels)
        plt.ylabel("Count")
        plt.title("Class Distribution - SPR Test Set")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_class_distribution.png"))
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()
