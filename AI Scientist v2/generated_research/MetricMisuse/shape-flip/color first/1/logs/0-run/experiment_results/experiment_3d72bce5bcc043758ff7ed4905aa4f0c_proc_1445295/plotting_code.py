import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- load data -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR" in experiment_data:
    run = experiment_data["SPR"]
    tr_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    epochs = run["epochs"]
    # unpack validation metrics
    cwa = [m["CWA"] for m in run["metrics"]["val"]]
    swa = [m["SWA"] for m in run["metrics"]["val"]]
    hpa = [m["HPA"] for m in run["metrics"]["val"]]
    # test label dists
    preds = np.array(run["predictions"])
    gts = np.array(run["ground_truth"])

    # -------- Fig 1: loss curves ----------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        ax[0].plot(epochs, tr_loss, label="Train")
        ax[1].plot(epochs, val_loss, label="Validation", color="orange")
        ax[0].set_title("Left: Train Loss")
        ax[1].set_title("Right: Validation Loss")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("Cross-Entropy Loss")
            a.legend()
        fig.suptitle("SPR Dataset – Training vs Validation Loss")
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # -------- Fig 2: weighted accuracies ----------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        ax[0].plot(epochs, cwa, label="CWA")
        ax[0].plot(epochs, swa, label="SWA")
        ax[1].plot(epochs, hpa, label="HPA", color="green")
        ax[0].set_title("Left: CWA & SWA")
        ax[1].set_title("Right: Harmonic Poly Accuracy")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("Score")
            a.legend()
        fig.suptitle("SPR Dataset – Validation Weighted Accuracies")
        fname = os.path.join(working_dir, "SPR_weighted_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating weighted accuracy curves: {e}")
        plt.close()

    # -------- Fig 3: test label distribution ----------
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)
        labels = sorted(list(set(gts)))
        gt_counts = [(gts == l).sum() for l in labels]
        pred_counts = [(preds == l).sum() for l in labels]
        width = 0.35
        x = np.arange(len(labels))
        ax.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        ax.bar(x + width / 2, pred_counts, width, label="Predictions")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Class Label")
        ax.set_ylabel("Count")
        ax.set_title(
            "Test Set Label Frequencies – SPR\nLeft: Ground Truth, Right: Predictions"
        )
        ax.legend()
        fname = os.path.join(working_dir, "SPR_test_label_distribution.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating label distribution plot: {e}")
        plt.close()
