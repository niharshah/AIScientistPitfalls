import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# set up paths ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# load data ---------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["RemoveRecurrent"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    epochs = exp.get("epochs", [])
    c_loss = exp.get("losses", {}).get("contrastive", [])
    tr_loss = exp.get("losses", {}).get("train_sup", [])
    val_loss = exp.get("losses", {}).get("val_sup", [])
    acs = exp.get("metrics", {}).get("val_ACS", [])
    preds = np.array(exp.get("predictions", []))
    gts = np.array(exp.get("ground_truth", []))

    # --------------------------------------------------------
    # 1. contrastive loss ------------------------------------
    try:
        if c_loss:
            plt.figure()
            plt.plot(range(1, len(c_loss) + 1), c_loss, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("SPR: Contrastive Pre-training Loss")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_contrastive_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating contrastive plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 2. supervised losses -----------------------------------
    try:
        if tr_loss and val_loss:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train", marker="o")
            plt.plot(epochs, val_loss, label="Validation", marker="s")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("SPR: Supervised Loss Curves")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_supervised_losses.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating supervised loss plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 3. augmentation consistency ----------------------------
    try:
        if acs:
            plt.figure()
            plt.plot(epochs, acs, marker="^", color="purple")
            plt.xlabel("Epoch")
            plt.ylabel("ACS")
            plt.title("SPR: Augmentation Consistency Score (Validation)")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_val_ACS.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating ACS plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 4. prediction vs ground truth distribution -------------
    try:
        if preds.size and gts.size:
            n_classes = max(preds.max(), gts.max()) + 1
            ind = np.arange(n_classes)
            plt.figure()
            plt.bar(
                ind - 0.15,
                np.bincount(gts, minlength=n_classes),
                width=0.3,
                label="Ground Truth",
            )
            plt.bar(
                ind + 0.15,
                np.bincount(preds, minlength=n_classes),
                width=0.3,
                label="Predictions",
            )
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(
                "SPR: Class Distribution (Last Epoch)\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_class_distribution.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # print final metrics ------------------------------------
    if preds.size and gts.size:
        accuracy = (preds == gts).mean()
        print(f"Final Validation Accuracy: {accuracy:.4f}")
    if acs:
        print(f"Final Augmentation Consistency Score: {acs[-1]:.4f}")
