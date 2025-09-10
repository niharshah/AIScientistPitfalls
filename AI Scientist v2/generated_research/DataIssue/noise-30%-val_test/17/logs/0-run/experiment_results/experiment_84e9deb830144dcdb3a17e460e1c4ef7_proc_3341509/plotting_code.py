import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["no_pos_embedding"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    # helpers
    def mcc(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom

    train_losses = np.asarray(exp["losses"]["train"])
    val_losses = np.asarray(exp["losses"]["val"])
    train_mccs = np.asarray(exp["metrics"]["train"])
    val_mccs = np.asarray(exp["metrics"]["val"])

    # ---------- 1) loss curves ----------
    try:
        plt.figure()
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH - Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------- 2) MCC curves ----------
    try:
        plt.figure()
        epochs = np.arange(1, len(train_mccs) + 1)
        plt.plot(epochs, train_mccs, label="Train")
        plt.plot(epochs, val_mccs, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("SPR_BENCH - Training vs Validation MCC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_MCC_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curves: {e}")
        plt.close()

    # ---------- 3) bar chart of test MCC per run ----------
    try:
        test_mccs = []
        for p, g in zip(exp["predictions"], exp["ground_truth"]):
            test_mccs.append(mcc(np.asarray(g), np.asarray(p)))
        plt.figure()
        idx = np.arange(len(test_mccs))
        plt.bar(idx, test_mccs, color="skyblue")
        plt.xlabel("Run Index")
        plt.ylabel("Test MCC")
        plt.title("SPR_BENCH - Test MCC per Run")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_MCC_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC bar chart: {e}")
        plt.close()

    # ---------- 4) confusion matrix for best run ----------
    try:
        best_idx = int(np.argmax(test_mccs))
        preds = np.asarray(exp["predictions"][best_idx]).astype(int)
        gts = np.asarray(exp["ground_truth"][best_idx]).astype(int)
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
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
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(
            f"SPR_BENCH - Confusion Matrix (Best Run {best_idx})\n"
            "Rows: Ground Truth, Columns: Prediction"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_confusion_matrix_best_run.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
