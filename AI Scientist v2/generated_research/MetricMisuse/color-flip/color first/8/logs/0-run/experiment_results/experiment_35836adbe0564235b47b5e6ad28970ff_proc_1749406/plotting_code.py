import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["train_only_kmeans"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    # 1) LOSS CURVES -------------------------------------------------
    try:
        plt.figure()
        for split in ["train", "val"]:
            if exp["losses"][split]:
                epochs, losses = zip(*exp["losses"][split])
                plt.plot(epochs, losses, label=f"{split} loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2) VALIDATION METRICS -----------------------------------------
    try:
        if exp["metrics"]["val"]:
            epochs, cwa, swa, hm, ocga = zip(*exp["metrics"]["val"])
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, hm, label="HM")
            plt.plot(epochs, ocga, label="OCGA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("SPR Validation Metrics")
            plt.legend()
            fname = os.path.join(working_dir, "SPR_validation_metrics.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # 3) CONFUSION MATRIX -------------------------------------------
    try:
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        if preds.size and gts.size:
            n_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                "SPR Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
