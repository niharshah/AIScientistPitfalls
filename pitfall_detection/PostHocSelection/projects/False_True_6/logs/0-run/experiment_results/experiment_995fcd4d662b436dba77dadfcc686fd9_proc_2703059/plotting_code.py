import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load artefacts ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_HYBRID" in experiment_data:
    run = experiment_data["SPR_HYBRID"]
    train_losses = run["losses"]["train"]
    val_losses = run["losses"]["val"]
    val_swa_hist = [m["SWA"] for m in run["metrics"]["val"]]
    test_swa = run["metrics"]["test"].get("SWA", None)
    preds = np.array(run["predictions"])
    trues = np.array(run["ground_truth"])
    labels = sorted(set(trues) | set(preds))
    n_epochs = range(1, len(train_losses) + 1)

    # ---------- 1) loss curves ----------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(n_epochs, train_losses, "b-o", label="Train Loss")
        plt.plot(n_epochs, val_losses, "r-o", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("SPR_HYBRID Loss Curves\nLeft: Train, Right: Validation")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_HYBRID_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- 2) validation SWA ----------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(n_epochs, val_swa_hist, "g-s")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.title("SPR_HYBRID Validation SWA over Epochs")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_HYBRID_val_SWA.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()

    # ---------- 3) test SWA bar ----------
    try:
        if test_swa is not None:
            plt.figure(figsize=(4, 3))
            plt.bar(["Test"], [test_swa], color="coral")
            plt.ylim(0, 1)
            plt.ylabel("SWA")
            plt.title("SPR_HYBRID Test SWA")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_HYBRID_test_SWA.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating test SWA bar: {e}")
        plt.close()

    # ---------- 4) confusion matrix ----------
    try:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted"), plt.ylabel("Ground Truth")
        plt.title("SPR_HYBRID Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.xticks(labels)
        plt.yticks(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_HYBRID_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- print metric ----------
    if test_swa is not None:
        print(f"Test SWA: {test_swa:.4f}")
else:
    print("SPR_HYBRID run not found in experiment_data.")
