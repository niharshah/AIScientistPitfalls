import matplotlib.pyplot as plt
import numpy as np
import os

# working directory -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data --------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp = experiment_data.get("multi_syn_pretrain", {}).get("SPR", {})


# helper to maybe subsample long curves ---------------------------------------
def limit_points(xs, ys, max_pts=5):
    if len(xs) <= max_pts:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, max_pts, dtype=int)
    return [xs[i] for i in idx], [ys[i] for i in idx]


# 1) contrastive pre-training loss --------------------------------------------
try:
    losses = exp.get("losses", {}).get("contrastive", [])
    if losses:
        epochs = list(range(1, len(losses) + 1))
        epochs, losses = limit_points(epochs, losses)
        plt.figure()
        plt.plot(epochs, losses, marker="o", label="Contrastive Loss")
        plt.xlabel("Pre-training Epoch")
        plt.ylabel("Loss")
        plt.title("SPR: Contrastive Pre-training Loss Curve")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_contrastive_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating contrastive loss plot: {e}")
    plt.close()

# 2) supervised train/val loss -------------------------------------------------
try:
    tr = exp.get("losses", {}).get("train_sup", [])
    vl = exp.get("losses", {}).get("val_sup", [])
    if tr and vl:
        epochs = exp.get("epochs", list(range(1, len(tr) + 1)))
        epochs, tr = limit_points(epochs, tr)
        _, vl = limit_points(epochs, vl)  # same indices as epochs
        plt.figure()
        plt.plot(epochs, tr, marker="o", label="Train Loss")
        plt.plot(epochs, vl, marker="s", label="Val Loss")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Loss")
        plt.title("SPR: Supervised Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_supervised_losses.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating supervised loss plot: {e}")
    plt.close()

# 3) validation ACS ------------------------------------------------------------
try:
    acs = exp.get("metrics", {}).get("val_ACS", [])
    if acs:
        epochs = exp.get("epochs", list(range(1, len(acs) + 1)))
        epochs, acs = limit_points(epochs, acs)
        plt.figure()
        plt.plot(epochs, acs, marker="^", color="green", label="Val ACS")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Augmentation Consistency Score")
        plt.title("SPR: Validation Augmentation Consistency Score")
        plt.ylim(0, 1)
        plt.legend()
        fname = os.path.join(working_dir, "SPR_val_ACS.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating ACS plot: {e}")
    plt.close()
