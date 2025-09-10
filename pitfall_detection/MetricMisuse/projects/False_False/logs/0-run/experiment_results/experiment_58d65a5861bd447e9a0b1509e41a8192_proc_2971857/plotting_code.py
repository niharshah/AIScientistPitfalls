import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["supervised_only"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    losses_train = ed["losses"]["train"]
    losses_val = ed["losses"]["val"]
    m_swa = ed["metrics"]["val_SWA"]
    m_cwa = ed["metrics"]["val_CWA"]
    m_scwa = ed["metrics"]["val_SCWA"]
    preds = np.array(ed["predictions"])
    trues = np.array(ed["ground_truth"])
    epochs = np.arange(1, len(losses_train) + 1)
    best_ep = int(np.argmax(m_scwa))
    best_scwa = m_scwa[best_ep]

    # ---------- PLOT 1: loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, losses_train, label="Train Loss")
        plt.plot(epochs, losses_val, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Training vs. Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- PLOT 2: validation metrics ----------
    try:
        plt.figure()
        plt.plot(epochs, m_swa, label="Shape-Weighted Acc.")
        plt.plot(epochs, m_cwa, label="Color-Weighted Acc.")
        plt.plot(epochs, m_scwa, label="Combined SCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Validation Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_metrics_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics curve: {e}")
        plt.close()

    # ---------- PLOT 3: confusion matrix ----------
    try:
        from itertools import product

        num_labels = len(np.unique(trues))
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix (Dev Set)")
        # write counts
        for i, j in product(range(num_labels), range(num_labels)):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------- print evaluation summary ----------
    accuracy = (preds == trues).mean() if len(preds) else 0.0
    print(f"Best epoch: {best_ep + 1}")
    print(f"Accuracy at best epoch: {accuracy:.4f}")
    print(f"Best SCWA: {best_scwa:.4f}")
