import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["no_variety_stats"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # ------------ loss curves ----------------
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("spr_bench Loss Curves\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ----------- SWA curves ------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train_swa"], label="Train SWA")
        plt.plot(epochs, ed["metrics"]["val_swa"], label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("spr_bench SWA Curves\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ----------- test confusion --------------
    try:
        g = np.array(ed["ground_truth"])
        p = np.array(ed["predictions"])
        cm = np.zeros((2, 2), dtype=int)
        for gt, pr in zip(g, p):
            cm[gt, pr] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
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
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        plt.title("spr_bench Test Confusion Matrix\nLeft-Top: TN, Right-Bottom: TP")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
