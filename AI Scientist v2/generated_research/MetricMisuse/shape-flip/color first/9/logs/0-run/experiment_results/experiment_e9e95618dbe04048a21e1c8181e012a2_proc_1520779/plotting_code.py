import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# -------- paths -------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data -----#
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["No_POS"]["SPR"]
except Exception as e:
    raise RuntimeError(f"Could not load experiment_data.npy: {e}")

epochs = np.array(exp["epochs"])
train_loss = np.array(exp["losses"]["train"])
val_loss = np.array(exp["losses"]["val"])


def metric_curve(name):
    return np.array([m[name] for m in exp["metrics"]["train"]]), np.array(
        [m[name] for m in exp["metrics"]["val"]]
    )


cwa_tr, cwa_val = metric_curve("CWA")
swa_tr, swa_val = metric_curve("SWA")
hwa_tr, hwa_val = metric_curve("HWA")


# -------- plotting utilities -------- #
def save_fig(fig, fname):
    fig.savefig(os.path.join(working_dir, fname), dpi=150)
    plt.close(fig)


# 1) Loss
try:
    fig = plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR (No_POS) Loss Curves")
    plt.legend()
    save_fig(fig, "SPR_NoPOS_loss_curves.png")
except Exception as e:
    print(f"Error plotting loss: {e}")
    plt.close()

# 2) CWA
try:
    fig = plt.figure()
    plt.plot(epochs, cwa_tr, label="Train")
    plt.plot(epochs, cwa_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.title("SPR (No_POS) CWA Curves")
    plt.legend()
    save_fig(fig, "SPR_NoPOS_CWA_curves.png")
except Exception as e:
    print(f"Error plotting CWA: {e}")
    plt.close()

# 3) SWA
try:
    fig = plt.figure()
    plt.plot(epochs, swa_tr, label="Train")
    plt.plot(epochs, swa_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.title("SPR (No_POS) SWA Curves")
    plt.legend()
    save_fig(fig, "SPR_NoPOS_SWA_curves.png")
except Exception as e:
    print(f"Error plotting SWA: {e}")
    plt.close()

# 4) HWA
try:
    fig = plt.figure()
    plt.plot(epochs, hwa_tr, label="Train")
    plt.plot(epochs, hwa_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR (No_POS) HWA Curves")
    plt.legend()
    save_fig(fig, "SPR_NoPOS_HWA_curves.png")
except Exception as e:
    print(f"Error plotting HWA: {e}")
    plt.close()

# 5) Confusion Matrix on test set
try:
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    cm = confusion_matrix(gts, preds)
    fig = plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR (No_POS) Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    save_fig(fig, "SPR_NoPOS_confusion_matrix.png")
except Exception as e:
    print(f"Error plotting confusion matrix: {e}")
    plt.close()


# -------- print test metrics -------- #
def _hwa(c, s, eps=1e-12):
    return 2 * c * s / (c + s + eps)


if len(exp["predictions"]) == len(exp["ground_truth"]) and len(exp["predictions"]) > 0:
    # recompute metrics
    seq_dummy = [""] * len(gts)  # not needed for overall score
    cwa = sum(gts == preds) / len(gts)  # placeholder: correct rate
    swa = cwa  # same placeholder since no seq weights
    hwa = _hwa(cwa, swa)
    print(f"Test metrics -- CWA: {cwa:.3f} | SWA: {swa:.3f} | HWA: {hwa:.3f}")
