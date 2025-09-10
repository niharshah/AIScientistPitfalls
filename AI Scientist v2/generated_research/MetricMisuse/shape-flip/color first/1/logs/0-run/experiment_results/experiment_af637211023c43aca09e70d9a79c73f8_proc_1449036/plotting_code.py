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

if experiment_data is not None and "SPR" in experiment_data:
    spr = experiment_data["SPR"]
    tr_loss = spr["losses"]["train"]
    val_loss = spr["losses"]["val"]
    metrics_val = spr["metrics"]["val"]
    epochs = np.arange(1, len(tr_loss) + 1)

    # helpers
    def mlist(field):
        return [m[field] for m in metrics_val]

    # -------- Plot 1: Train vs Val loss -------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR – Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- Plot 2: Accuracy vs HPA -------------
    try:
        plt.figure()
        plt.plot(epochs, mlist("acc"), label="Accuracy")
        plt.plot(epochs, mlist("HPA"), label="Harmonic Poly Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR – Validation Accuracy vs HPA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_acc_hpa.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating acc/HPA plot: {e}")
        plt.close()

    # -------- Plot 3: CWA vs SWA -------------
    try:
        plt.figure()
        plt.plot(epochs, mlist("CWA"), label="CWA")
        plt.plot(epochs, mlist("SWA"), label="SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR – Color vs Shape Weighted Acc")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_cwa_swa.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating CWA/SWA plot: {e}")
        plt.close()

    # -------- Plot 4: Test label distribution -------------
    try:
        preds = np.array(spr["predictions"])
        gts = np.array(spr["ground_truth"])
        classes = sorted(set(gts.tolist() + preds.tolist()))
        width = 0.35
        plt.figure()
        plt.bar(
            np.arange(len(classes)) - width / 2,
            [np.sum(gts == c) for c in classes],
            width,
            label="Ground Truth",
        )
        plt.bar(
            np.arange(len(classes)) + width / 2,
            [np.sum(preds == c) for c in classes],
            width,
            label="Predictions",
        )
        plt.xticks(classes)
        plt.xlabel("Class label")
        plt.ylabel("Count")
        plt.title("SPR – Test Set: Ground Truth vs Predictions")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_test_distribution.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()
