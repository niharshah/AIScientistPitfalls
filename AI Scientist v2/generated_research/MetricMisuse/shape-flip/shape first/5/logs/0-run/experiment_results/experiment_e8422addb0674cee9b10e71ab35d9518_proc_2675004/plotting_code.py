import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def confusion_matrix(preds, gts, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, g in zip(preds, gts):
        cm[g, p] += 1
    return cm


try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset, dval in experiment_data.items():
    losses = dval.get("losses", {})
    metrics = dval.get("metrics", {})
    preds = dval.get("predictions")
    gts = dval.get("ground_truth")

    # 1) loss curves ----------------------------------------------------------
    try:
        if losses:
            plt.figure()
            if "train" in losses and len(losses["train"]):
                plt.plot(losses["train"], label="Train")
            if "val" in losses and len(losses["val"]):
                plt.plot(losses["val"], label="Validation")
            plt.title(f"{dset}: Training vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dset}: {e}")
        plt.close()

    # 2) metric curves --------------------------------------------------------
    try:
        if metrics:
            plt.figure()
            for mname, mvals in metrics.items():
                if len(mvals):
                    plt.plot(mvals, label=mname)
            plt.title(f"{dset}: Metric Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_metric_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting metrics for {dset}: {e}")
        plt.close()

    # 3) confusion matrix / accuracy -----------------------------------------
    try:
        if preds is not None and gts is not None:
            preds = np.asarray(preds)
            gts = np.asarray(gts)
            acc = (preds == gts).mean()
            print(f"{dset}: Test accuracy = {acc:.4f}")
            num_classes = int(max(preds.max(), gts.max()) + 1)
            if num_classes <= 5:  # keep plot readable
                cm = confusion_matrix(preds, gts, num_classes)
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.title(f"{dset}: Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.colorbar()
                fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
                plt.savefig(fname)
                plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset}: {e}")
        plt.close()
