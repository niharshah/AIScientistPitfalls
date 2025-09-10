import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ed = experiment_data["pooling_last_token"]["SPR_BENCH"]
    epochs = ed["epochs"]

    # -------- helper ---------
    def save_plot(fig, fname):
        fig.savefig(os.path.join(working_dir, fname))
        plt.close(fig)

    # -------- loss curve --------
    try:
        fig = plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="train")
        plt.plot(epochs, ed["losses"]["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curve (Pooling-Last-Token)")
        plt.legend()
        save_plot(fig, "SPR_BENCH_loss_curve.png")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- metrics curves --------
    metric_names = ["CWA", "SWA", "CplxWA"]
    for m in metric_names:
        try:
            fig = plt.figure()
            plt.plot(epochs, ed["metrics"]["train"][m], label=f"train {m}")
            plt.plot(epochs, ed["metrics"]["val"][m], label=f"val {m}")
            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.title(f"SPR_BENCH {m} Curve (Pooling-Last-Token)")
            plt.legend()
            save_plot(fig, f"SPR_BENCH_{m}_curve.png")
        except Exception as e:
            print(f"Error creating {m} plot: {e}")
            plt.close()

    # -------- confusion matrix --------
    try:
        gt = np.array(ed["ground_truth"])
        pred = np.array(ed["predictions"])
        classes = np.sort(np.unique(gt))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for g, p in zip(gt, pred):
            cm[g, p] += 1

        fig = plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("SPR_BENCH Confusion Matrix (Test Set)")
        plt.xticks(classes)
        plt.yticks(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        save_plot(fig, "SPR_BENCH_confusion_matrix.png")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -------- print final test metrics --------
    print(
        "Final Test Metrics:",
        {k: round(v, 4) for k, v in ed["metrics"]["test"].items()},
    )
