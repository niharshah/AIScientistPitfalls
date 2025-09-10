import matplotlib.pyplot as plt
import numpy as np
import os

# set working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    node = exp["dual_encoder_no_share"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    node = None

if node:
    train_loss = node["losses"]["train"]
    val_loss = node["losses"]["val"]
    val_metrics = node["metrics"]["val"]  # list of dicts
    epochs = range(1, len(train_loss) + 1)

    # ------------- Loss curve -------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("spr_bench Loss Curve\nLeft: Training, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------- Metric curves ----------
    try:
        swa = [m["swa"] for m in val_metrics]
        cwa = [m["cwa"] for m in val_metrics]
        ccwa = [m["ccwa"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, ccwa, label="CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("spr_bench Validation Metrics\nSWA / CWA / CCWA vs Epoch")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_metric_curves.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # ------------- Confusion matrix -------
    try:
        preds = node.get("predictions", [])
        gts = node.get("ground_truth", [])
        if preds and gts:
            num_lab = max(max(preds), max(gts)) + 1
            cm = np.zeros((num_lab, num_lab), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                "spr_bench Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            for i in range(num_lab):
                for j in range(num_lab):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
            print("Saved", fname)
        else:
            print(
                "Prediction/ground-truth arrays not found, skipping confusion matrix."
            )
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------- print final val metrics -----
    if val_metrics:
        last = val_metrics[-1]
        print(
            f"Final Val -> Epoch {last['epoch']}: "
            f"SWA={last['swa']:.4f}, CWA={last['cwa']:.4f}, CCWA={last['ccwa']:.4f}, Loss={last['loss']:.4f}"
        )
