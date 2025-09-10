import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_exp = experiment_data.get("freeze_encoder", {}).get("SPR", {})


# helper ------------------------------------------------
def save_fig(fig, fname):
    fig.savefig(os.path.join(working_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# 1) contrastive loss -----------------------------------
try:
    losses = spr_exp.get("losses", {})
    contr = losses.get("contrastive", [])
    if contr:
        fig = plt.figure()
        plt.plot(range(1, len(contr) + 1), contr, marker="o")
        plt.title("SPR dataset – Contrastive Stage Loss")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        save_fig(fig, "SPR_contrastive_loss.png")
except Exception as e:
    print(f"Error creating contrastive plot: {e}")
    plt.close()

# 2) supervised loss ------------------------------------
try:
    train_sup = losses.get("train_sup", [])
    val_sup = losses.get("val_sup", [])
    if train_sup and val_sup:
        fig = plt.figure()
        epochs = spr_exp.get("epochs", list(range(1, len(train_sup) + 1)))
        plt.plot(epochs, train_sup, label="Train")
        plt.plot(epochs, val_sup, label="Validation")
        plt.title("SPR dataset – Supervised Loss (Train vs Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        save_fig(fig, "SPR_supervised_loss.png")
except Exception as e:
    print(f"Error creating supervised loss plot: {e}")
    plt.close()

# 3) validation accuracy --------------------------------
try:
    val_acc = spr_exp.get("metrics", {}).get("val_acc", [])
    if val_acc:
        fig = plt.figure()
        epochs = spr_exp.get("epochs", list(range(1, len(val_acc) + 1)))
        plt.plot(epochs, val_acc, marker="s", color="green")
        plt.ylim(0, 1)
        plt.title("SPR dataset – Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        save_fig(fig, "SPR_val_accuracy.png")
except Exception as e:
    print(f"Error creating val accuracy plot: {e}")
    plt.close()

# 4) augmentation-consistency ---------------------------
try:
    m = spr_exp.get("metrics", {})
    tr_acs = m.get("train_ACS", [])
    val_acs = m.get("val_ACS", [])
    if tr_acs and val_acs:
        fig = plt.figure()
        epochs = spr_exp.get("epochs", list(range(1, len(tr_acs) + 1)))
        plt.plot(epochs, tr_acs, label="Train ACS")
        plt.plot(epochs, val_acs, label="Val ACS")
        plt.ylim(0, 1)
        plt.title("SPR dataset – Augmentation Consistency Score")
        plt.xlabel("Epoch")
        plt.ylabel("ACS")
        plt.legend()
        save_fig(fig, "SPR_ACS.png")
except Exception as e:
    print(f"Error creating ACS plot: {e}")
    plt.close()

# ---------------- print evaluation metric ---------------
try:
    if val_acc:
        print(f"Final validation accuracy: {val_acc[-1]:.4f}")
except Exception as e:
    print(f"Error printing evaluation metric: {e}")
