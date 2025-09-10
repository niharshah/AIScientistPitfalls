import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ed = experiment_data["HIDDEN_DIM"]["SPR_BENCH"]
    hiddens = np.array([hp["HIDDEN_DIM"] for hp in ed["hparams"]])
    train_loss = np.array(ed["metrics"]["train_loss"])
    val_loss = np.array(ed["metrics"]["val_loss"])
    val_bps = np.array(ed["metrics"]["val_bps"])
    val_swa = np.array(ed["metrics"]["val_swa"])
    val_cwa = np.array(ed["metrics"]["val_cwa"])

    # 1. Train / Val loss curve
    try:
        plt.figure()
        plt.plot(hiddens, train_loss, marker="o", label="Train Loss")
        plt.plot(hiddens, val_loss, marker="s", label="Validation Loss")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs. Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_vs_hidden_dim.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. BPS bar plot
    try:
        plt.figure()
        plt.bar(hiddens.astype(str), val_bps)
        plt.xlabel("Hidden Dimension")
        plt.ylabel("BPS")
        plt.title("SPR_BENCH: Validation BPS by Hidden Dimension")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_bps_by_hidden_dim.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating BPS plot: {e}")
        plt.close()

    # 3. SWA vs CWA scatter
    try:
        plt.figure()
        plt.scatter(val_swa, val_cwa)
        for x, y, h in zip(val_swa, val_cwa, hiddens):
            plt.annotate(str(h), (x, y))
        plt.xlabel("Shape-Weighted Accuracy (SWA)")
        plt.ylabel("Color-Weighted Accuracy (CWA)")
        plt.title("SPR_BENCH: SWA vs. CWA (numbers = hidden dim)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_swa_vs_cwa.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA/CWA scatter: {e}")
        plt.close()

    # 4. Confusion matrix for best model
    try:
        best_idx = int(np.argmax(val_bps))
        preds = np.array(ed["predictions"]["dev"][best_idx])
        labels = np.array(ed["ground_truth"]["dev"][best_idx])
        num_classes = int(max(preds.max(), labels.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(labels, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"SPR_BENCH Confusion Matrix (Hidden={hiddens[best_idx]})")
        plt.tight_layout()
        fname = os.path.join(
            working_dir, f"SPR_BENCH_confusion_matrix_hidden_{hiddens[best_idx]}.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
