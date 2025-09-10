import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# ---- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# short-circuit if load failed
if experiment_data:
    run = experiment_data["FrozenEmbeddingLayer"]["SPR_BENCH"]
    tr_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    tr_mcc = run["metrics"]["train_MCC"]
    val_mcc = run["metrics"]["val_MCC"]
    preds = np.array(run["predictions"])
    gts = np.array(run["ground_truth"])
    epochs = np.arange(1, len(tr_loss) + 1)

    # ---- plot 1: loss curves ----
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss Curves – SPR_BENCH (FrozenEmbeddingLayer)")
        plt.legend()
        fname = os.path.join(
            working_dir, "FrozenEmbeddingLayer_SPR_BENCH_loss_curves.png"
        )
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---- plot 2: MCC curves ----
    try:
        plt.figure()
        plt.plot(epochs, tr_mcc, label="Train MCC")
        plt.plot(epochs, val_mcc, label="Val MCC")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("MCC Curves – SPR_BENCH (FrozenEmbeddingLayer)")
        plt.legend()
        fname = os.path.join(
            working_dir, "FrozenEmbeddingLayer_SPR_BENCH_MCC_curves.png"
        )
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating MCC curve plot: {e}")
        plt.close()

    # ---- plot 3: confusion matrix ----
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, v, ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix – SPR_BENCH Test Set")
        fname = os.path.join(
            working_dir, "FrozenEmbeddingLayer_SPR_BENCH_confusion_matrix.png"
        )
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---- print stored metrics ----
    best_val_mcc = max(val_mcc) if val_mcc else None
    from sklearn.metrics import matthews_corrcoef, f1_score

    test_mcc = matthews_corrcoef(gts, preds) if preds.size else None
    test_f1 = f1_score(gts, preds, average="macro") if preds.size else None
    print(
        f"Best Val MCC={best_val_mcc:.4f} | Test MCC={test_mcc:.4f} | Test F1={test_f1:.4f}"
    )
