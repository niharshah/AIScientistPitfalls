import matplotlib.pyplot as plt
import numpy as np
import os

# set / make working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# short-circuit if data missing
if not experiment_data:
    print("No experiment data found – nothing to plot.")
else:
    exp = experiment_data["no_ffn"]["SPR"]

    # 1) BCE-loss curves -------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(exp["losses"]["train"], label="train")
        plt.plot(exp["losses"]["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("BCE loss")
        plt.legend()
        plt.title("SPR – No-FFN Transformer\nTraining vs Validation Loss")
        fname = os.path.join(working_dir, "SPR_no_ffn_loss_curve.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) MCC curves -----------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(exp["metrics"]["train_MCC"], label="train")
        plt.plot(exp["metrics"]["val_MCC"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        plt.title("SPR – No-FFN Transformer\nTraining vs Validation MCC")
        fname = os.path.join(working_dir, "SPR_no_ffn_MCC_curve.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # 3) Confusion matrix heat-map -------------------------------------------
    try:
        # compute 2×2 confusion matrix
        preds = np.array(exp["predictions"], dtype=int)
        gts = np.array(exp["ground_truth"], dtype=int)
        if preds.size and gts.size:
            cm = np.zeros((2, 2), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1  # rows: truth, cols: pred
            plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
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
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.title("SPR – No-FFN Transformer\nTest Confusion Matrix")
            fname = os.path.join(working_dir, "SPR_no_ffn_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
            print(f"Test MCC={exp['test_MCC']:.3f} | Macro-F1={exp['test_F1']:.3f}")
        else:
            print("Prediction / ground-truth arrays empty – skipping confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
