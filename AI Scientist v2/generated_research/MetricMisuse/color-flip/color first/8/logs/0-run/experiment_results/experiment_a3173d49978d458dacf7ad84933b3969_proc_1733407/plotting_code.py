import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

ds_key = "SPR"  # dataset name used during training

# ---------------- Figure 1: loss curves -------------------------
try:
    if experiment_data is None or ds_key not in experiment_data:
        raise ValueError("Experiment data missing required key.")
    plt.figure()
    # training loss
    tr = experiment_data[ds_key]["losses"]["train"]
    if tr:
        e, l = zip(*tr)
        plt.plot(e, l, "--", label="train loss")
    # validation loss
    vl = experiment_data[ds_key]["losses"]["val"]
    if vl:
        e, l = zip(*vl)
        plt.plot(e, l, "-", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR dataset – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print("Saved", fname)
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------- Figure 2: CWA & SWA ---------------------------
try:
    plt.figure()
    vm = experiment_data[ds_key]["metrics"]["val"]
    if not vm:
        raise ValueError("No validation metrics.")
    epochs = [t[0] for t in vm]
    cwa = [t[1] for t in vm]
    swa = [t[2] for t in vm]
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR – Validation Colour vs Shape Weighted Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "spr_cwa_swa_curves.png")
    plt.savefig(fname)
    plt.close()
    print("Saved", fname)
except Exception as e:
    print(f"Error creating CWA/SWA plot: {e}")
    plt.close()

# ---------------- Figure 3: HM & OCGA ---------------------------
try:
    plt.figure()
    hm = [t[3] for t in vm]
    ocga = [t[4] for t in vm]
    plt.plot(epochs, hm, label="HM")
    plt.plot(epochs, ocga, label="OCGA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR – Harmonic Mean (HM) & OCGA over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "spr_hm_ocga_curves.png")
    plt.savefig(fname)
    plt.close()
    print("Saved", fname)
except Exception as e:
    print(f"Error creating HM/OCGA plot: {e}")
    plt.close()

# ---------------- Figure 4: confusion matrix --------------------
try:
    from itertools import product

    gt = experiment_data[ds_key]["ground_truth"]
    pr = experiment_data[ds_key]["predictions"]
    if not gt or not pr:
        raise ValueError("Ground-truth / prediction lists empty.")
    gt = np.array(gt)
    pr = np.array(pr)
    num_cls = max(gt.max(), pr.max()) + 1
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for g, p in zip(gt, pr):
        cm[g, p] += 1
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("SPR – Confusion Matrix (GT rows vs Pred cols)")
    plt.xlabel("Predicted label")
    plt.ylabel("Ground-truth label")
    for i, j in product(range(num_cls), range(num_cls)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=7)
    fname = os.path.join(working_dir, "spr_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
    print("Saved", fname)
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- Figure 5: error histogram ---------------------
try:
    plt.figure()
    errors = (gt != pr).astype(int)
    plt.hist(errors, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1], ["Correct", "Incorrect"])
    plt.ylabel("Count")
    plt.title("SPR – Prediction Error Histogram")
    fname = os.path.join(working_dir, "spr_error_histogram.png")
    plt.savefig(fname)
    plt.close()
    print("Saved", fname)
except Exception as e:
    print(f"Error creating error histogram: {e}")
    plt.close()
