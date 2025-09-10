import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def macro_f1(preds, labels, num_cls):
    f1s = []
    for c in range(num_cls):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


# ------------------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["NoCLS_MeanPool"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = None

if run:
    epochs = np.arange(1, len(run["losses"]["train"]) + 1)

    # ------------------------- loss curves ------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, run["losses"]["train"], label="Train")
        plt.plot(epochs, run["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves (NoCLS_MeanPool)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_loss_curves_NoCLS_MeanPool.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------------- accuracy curves -------------------------------
    try:
        plt.figure()
        plt.plot(epochs, run["metrics"]["train_acc"], label="Train")
        plt.plot(epochs, run["metrics"]["val_acc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves (NoCLS_MeanPool)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_accuracy_curves_NoCLS_MeanPool.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------------------- F1 curves -------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, run["metrics"]["train_f1"], label="Train")
        plt.plot(epochs, run["metrics"]["val_f1"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curves (NoCLS_MeanPool)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves_NoCLS_MeanPool.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # ------------------------- confusion matrix ------------------------------
    try:
        preds = run["predictions"]
        gts = run["ground_truth"]
        if len(preds) and len(gts):
            classes = np.unique(np.concatenate([preds, gts]))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR_BENCH Confusion Matrix (Test set)")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, "SPR_BENCH_confusion_matrix_NoCLS_MeanPool.png"
                )
            )
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------------- print metrics ----------------------------------
    try:
        preds = run["predictions"]
        gts = run["ground_truth"]
        if len(preds) and len(gts):
            acc = (preds == gts).mean()
            f1 = macro_f1(preds, gts, len(np.unique(gts)))
            print(f"Test accuracy: {acc*100:.2f}% | Test macroF1: {f1:.4f}")
    except Exception as e:
        print(f"Error computing metrics: {e}")
