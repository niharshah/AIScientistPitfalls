import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

bench = experiment_data["SPR"]

# ----------------------------- 1) Loss curves -------------------------------
try:
    train_loss = bench["losses"]["train"]
    val_loss = bench["losses"]["val"]
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Synthetic: Training vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_synth_loss_curves.png")
    plt.savefig(fname, dpi=140)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ----------------------------- 2) Metric curves -----------------------------
try:
    swa = [m["swa"] for m in bench["metrics"]["val"]]
    cwa = [m["cwa"] for m in bench["metrics"]["val"]]
    acs = [m["acs"] for m in bench["metrics"]["val"]]
    epochs = np.arange(1, len(swa) + 1)

    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, acs, label="ACS")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR Synthetic: Weighted Accuracy Metrics over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "spr_synth_metric_curves.png")
    plt.savefig(fname, dpi=140)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ----------------------------- 3) Confusion matrix --------------------------
try:
    preds = np.array(bench["predictions"])
    gts = np.array(bench["ground_truth"])
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.colorbar(im)
        plt.xticks([0, 1], ["Class 0", "Class 1"])
        plt.yticks([0, 1], ["Class 0", "Class 1"])
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR Synthetic: Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "spr_synth_confusion_matrix.png")
        plt.savefig(fname, dpi=140)
        print("Saved", fname)
        plt.close()
    else:
        print("No prediction/ground-truth data found for confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
