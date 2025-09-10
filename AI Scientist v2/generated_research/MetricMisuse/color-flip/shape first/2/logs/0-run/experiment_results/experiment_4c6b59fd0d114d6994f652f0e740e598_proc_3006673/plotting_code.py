import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    node = experiment_data["no_contrastive"]["spr_bench"]
    train_loss = np.asarray(node["losses"]["train"])
    val_loss = np.asarray(node["losses"]["val"])
    swa_vals = np.asarray([d["swa"] for d in node["metrics"]["val"]])
    cwa_vals = np.asarray([d["cwa"] for d in node["metrics"]["val"]])
    ccwa_vals = np.asarray([d["ccwa"] for d in node["metrics"]["val"]])
    epochs = np.arange(1, len(train_loss) + 1)

    # ----------------- plot 1: loss curves -----------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves (No-Contrastive)\nTraining vs Validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "spr_bench_loss_curves_no_contrastive.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------------- plot 2: weighted accuracies -----------------
    try:
        plt.figure()
        plt.plot(epochs, swa_vals, label="SWA")
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.plot(epochs, ccwa_vals, label="CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Weighted Accuracies (No-Contrastive)\nSWA / CWA / CCWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "spr_bench_weighted_acc_no_contrastive.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ----------------- plot 3: confusion matrix -----------------
    preds = node.get("predictions", [])
    trues = node.get("ground_truth", [])
    if len(preds) and len(trues):
        try:
            preds = np.asarray(preds, dtype=int)
            trues = np.asarray(trues, dtype=int)
            num_labels = int(max(preds.max(), trues.max()) + 1)
            cm = np.zeros((num_labels, num_labels), dtype=int)
            for t, p in zip(trues, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                "SPR_BENCH Confusion Matrix (No-Contrastive)\nGround Truth vs Predictions"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, "spr_bench_confusion_matrix_no_contrastive.png"
                )
            )
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix plot: {e}")
            plt.close()

    # ----------------- print best CCWA -----------------
    if ccwa_vals.size:
        best_ccwa = ccwa_vals.max()
        print(f"Best validation CCWA: {best_ccwa:.4f}")
