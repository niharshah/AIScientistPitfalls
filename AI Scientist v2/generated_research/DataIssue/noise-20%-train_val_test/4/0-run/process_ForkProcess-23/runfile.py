import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    edict = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = edict["cls_pooling"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    epochs = ed["epochs"]
    tr_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    tr_f1 = ed["metrics"]["train_f1"]
    val_f1 = ed["metrics"]["val_f1"]
    test_f1 = ed["metrics"]["test_f1"]
    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])

    # 1. Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench Loss Curves\nLeft: Training, Right: Validation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. F1 curves
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        if test_f1 is not None:
            plt.hlines(
                test_f1,
                xmin=epochs[0],
                xmax=epochs[-1],
                colors="r",
                linestyles="dashed",
                label=f"Test={test_f1:.3f}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("spr_bench F1 Curves\nLeft: Training, Right: Validation")
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        fname = os.path.join(working_dir, "spr_bench_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # 3. Confusion matrix (optional)
    if preds and gts:
        try:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(
                "spr_bench Normalized Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()
