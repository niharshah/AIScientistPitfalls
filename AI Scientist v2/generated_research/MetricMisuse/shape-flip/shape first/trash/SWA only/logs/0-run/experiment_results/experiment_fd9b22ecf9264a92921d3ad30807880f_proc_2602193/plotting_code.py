import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ Load experiment data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data is not None:
    epochs = np.arange(1, len(data["metrics"]["train_acc"]) + 1)

    # ------------------ 1. Accuracy curve ------------------
    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, data["metrics"]["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Training vs Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------------ 2. Validation loss ------------------
    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["val_loss"], color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("SPR_BENCH: Validation Loss Across Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_val_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------ 3. NRGS bar ------------------
    try:
        nrg = data["metrics"]["NRGS"][0] if data["metrics"]["NRGS"] else None
        if nrg is not None:
            plt.figure()
            plt.bar(["NRGS"], [nrg], color="green")
            plt.ylim(0, 1)
            plt.title("SPR_BENCH: Novel Rule Generalization Score")
            fname = os.path.join(working_dir, "SPR_BENCH_NRGS.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating NRGS plot: {e}")
        plt.close()

    # ------------------ 4. Confusion matrix ------------------
    try:
        preds = np.array(data["predictions"])
        gts = np.array(data["ground_truth"])
        if preds.size and gts.size:
            classes = np.unique(np.concatenate([preds, gts]))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR_BENCH: Confusion Matrix")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------ Print final metrics ------------------
    final_acc = data["metrics"]["val_acc"][-1] if data["metrics"]["val_acc"] else None
    final_nrg = data["metrics"]["NRGS"][0] if data["metrics"]["NRGS"] else None
    print(f"Final Validation Accuracy: {final_acc}")
    print(f"Final NRGS: {final_nrg}")
