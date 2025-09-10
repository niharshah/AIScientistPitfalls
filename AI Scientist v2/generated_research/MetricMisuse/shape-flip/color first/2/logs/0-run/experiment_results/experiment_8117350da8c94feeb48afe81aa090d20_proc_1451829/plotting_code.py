import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def pcwa(seqs, y_true, y_pred):
    def cvar(s):
        return len(set(tok[1] for tok in s.strip().split() if len(tok) > 1))

    def svar(s):
        return len(set(tok[0] for tok in s.strip().split() if tok))

    w = np.array([cvar(s) * svar(s) for s in seqs], dtype=float)
    correct = (np.array(y_true) == np.array(y_pred)).astype(float)
    return ((w * correct).sum() / w.sum()) if w.sum() else 0.0


# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("UniChain", {}).get("SPR", {})
losses_train = ed.get("losses", {}).get("train", [])
losses_val = ed.get("losses", {}).get("val", [])
metrics_train = ed.get("metrics", {}).get("train", [])
metrics_val = ed.get("metrics", {}).get("val", [])
predictions = np.array(ed.get("predictions", []))
ground_truth = np.array(ed.get("ground_truth", []))
all_seqs = np.array(
    [None] * len(predictions)
)  # sequences not saved; pcwa will be skipped

# ---------------------------------------------------------------------
# 1) Loss curves -------------------------------------------------------
try:
    if losses_train and losses_val:
        ep_tr, val_tr = zip(*losses_train)
        ep_v, val_v = zip(*losses_val)
        plt.figure()
        plt.plot(ep_tr, val_tr, label="Train")
        plt.plot(ep_v, val_v, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("UniChain – SPR Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "unichain_spr_loss_curves.png")
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) PCWA metric curves ------------------------------------------------
try:
    if metrics_train and metrics_val:
        ep_tr, val_tr = zip(*metrics_train)
        ep_v, val_v = zip(*metrics_val)
        plt.figure()
        plt.plot(ep_tr, val_tr, label="Train PCWA")
        plt.plot(ep_v, val_v, label="Validation PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.title("UniChain – SPR PCWA Curves")
        plt.legend()
        fname = os.path.join(working_dir, "unichain_spr_pcwa_curves.png")
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating PCWA curve: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Confusion matrix --------------------------------------------------
try:
    if predictions.size and ground_truth.size:
        cm = np.zeros((2, 2), dtype=int)
        for gt, pr in zip(ground_truth, predictions):
            cm[gt, pr] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
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
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("UniChain – SPR Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "unichain_spr_confusion_matrix.png")
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------------------------------------------------------------
# Print evaluation metrics --------------------------------------------
if predictions.size and ground_truth.size:
    acc = (predictions == ground_truth).mean()
    pcwa_score = pcwa(
        all_seqs if all_seqs.any() else [""] * len(predictions),
        ground_truth,
        predictions,
    )
    print(f"Test ACC: {acc:.4f} | Test PCWA: {pcwa_score:.4f}")
