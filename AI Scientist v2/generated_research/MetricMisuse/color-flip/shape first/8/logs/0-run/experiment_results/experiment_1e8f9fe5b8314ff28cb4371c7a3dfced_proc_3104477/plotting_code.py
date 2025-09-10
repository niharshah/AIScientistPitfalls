import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment data ---------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    try:
        ed = experiment_data["FixedSinPE"]["SPR_BENCH"]
        train_losses = np.array(ed["losses"]["train"])
        val_losses = np.array(ed["losses"]["val"])
        val_ccwa = np.array(ed["metrics"]["val_CCWA"])
        preds_list = ed["predictions"]
        gts_list = ed["ground_truth"]
        epochs = np.arange(1, len(train_losses) + 1)
    except KeyError as e:
        print(f"Missing key while parsing data: {e}")
        ed, epochs = None, None

# ------------------ Plot 1: loss curves ----------------------
if ed is not None:
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------ Plot 2: CCWA metric ----------------------
    try:
        plt.figure()
        plt.plot(epochs, val_ccwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        plt.title("SPR_BENCH – Validation CCWA over Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_val_CCWA.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating CCWA plot: {e}")
        plt.close()

    # ------------------ Plot 3: confusion matrix last epoch -----
    try:
        y_true = np.array(gts_list[-1])
        y_pred = np.array(preds_list[-1])
        classes = np.arange(len(np.unique(np.concatenate([y_true, y_pred]))))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("SPR_BENCH – Confusion Matrix (Last Epoch)")
        plt.xticks(classes)
        plt.yticks(classes)
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_epoch_last.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
