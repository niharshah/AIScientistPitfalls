import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    spr_exp = experiment_data["learning_rate"]["SPR_BENCH"]
    runs = spr_exp["runs"]
    best_lr = spr_exp["best_lr"]
    preds = np.array(spr_exp["predictions"])
    gts = np.array(spr_exp["ground_truth"])

    # ---------- metric helper ----------
    def macro_f1(y_true, y_pred):
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for lb in labels:
            tp = np.sum((y_true == lb) & (y_pred == lb))
            fp = np.sum((y_true != lb) & (y_pred == lb))
            fn = np.sum((y_true == lb) & (y_pred != lb))
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            if prec + rec == 0:
                f1s.append(0.0)
            else:
                f1s.append(2 * prec * rec / (prec + rec))
        return np.mean(f1s)

    print("Test Macro-F1:", macro_f1(gts, preds))

    # ---------- 1) Val-F1 curves across learning rates ----------
    try:
        plt.figure()
        for lr, run in runs.items():
            val_f1 = run["metrics"]["val_f1"]
            epochs = np.arange(1, len(val_f1) + 1)
            plt.plot(epochs, val_f1, label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Macro-F1")
        plt.title("SPR_BENCH: Validation Macro-F1 vs Epoch (all LRs)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_val_f1_all_lrs.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val-F1 plot: {e}")
        plt.close()

    # ---------- 2) Train/Val loss for best_lr ----------
    try:
        run = runs[str(best_lr)]
        tr_loss = run["losses"]["train"]
        val_loss = run["losses"]["val"]
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH: Loss Curves (best lr={best_lr})")
        plt.legend()
        fname = os.path.join(working_dir, f"spr_loss_curves_best_lr_{best_lr}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 3) Confusion matrix ----------
    try:
        labels = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.xticks(labels)
        plt.yticks(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=7
                )
        fname = os.path.join(working_dir, "spr_confusion_matrix_test.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
