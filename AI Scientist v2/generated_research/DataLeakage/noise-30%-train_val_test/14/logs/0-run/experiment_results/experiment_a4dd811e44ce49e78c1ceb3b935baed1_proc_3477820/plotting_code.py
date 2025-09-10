import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    # helper to extract lists
    epochs_tr = [d["epoch"] for d in data["losses"]["train"]]
    train_loss = [d["loss"] for d in data["losses"]["train"]]
    val_loss = [d["loss"] for d in data["losses"]["val"]]
    train_f1 = [d["macro_f1"] for d in data["metrics"]["train"]]
    val_f1 = [d["macro_f1"] for d in data["metrics"]["val"]]
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))

    # ------------------------------------------------------------------ #
    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs_tr, train_loss, label="Train Loss")
        plt.plot(epochs_tr, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 2) Macro-F1 curves
    try:
        plt.figure()
        plt.plot(epochs_tr, train_f1, label="Train Macro-F1")
        plt.plot(epochs_tr, val_f1, label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating macro-F1 curve: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # 3) Confusion matrix
    try:
        if preds.size and gts.size:
            num_labels = len(np.unique(gts))
            cm = np.zeros((num_labels, num_labels), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(num_labels):
                for j in range(num_labels):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                "SPR_BENCH: Confusion Matrix (Dev Set)\nLeft: Ground Truth, Right: Predicted"
            )
            plt.colorbar()
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        else:
            print("No prediction/ground-truth data found; skipping confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------------------------------------------------------ #
    # Print final evaluation metrics
    try:
        from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

        if preds.size and gts.size:
            print("Final Dev-Set Metrics:")
            print("Macro-F1:", f1_score(gts, preds, average="macro"))
            print("Accuracy:", accuracy_score(gts, preds))
            print("MCC:", matthews_corrcoef(gts, preds))
    except Exception as e:
        print(f"Error computing evaluation metrics: {e}")
