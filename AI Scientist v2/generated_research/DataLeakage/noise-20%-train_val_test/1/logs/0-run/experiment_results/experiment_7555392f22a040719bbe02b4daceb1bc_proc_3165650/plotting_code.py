import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- #
# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# proceed only if data are present
if "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    # guard against missing keys
    tr_loss = np.array(data["losses"]["train"]) if data["losses"]["train"] else None
    val_loss = np.array(data["losses"]["val"]) if data["losses"]["val"] else None
    tr_macro = (
        np.array(data["metrics"]["train_macroF1"])
        if data["metrics"]["train_macroF1"]
        else None
    )
    val_macro = (
        np.array(data["metrics"]["val_macroF1"])
        if data["metrics"]["val_macroF1"]
        else None
    )
    preds = np.array(data["predictions"]) if len(data["predictions"]) else None
    gts = np.array(data["ground_truth"]) if len(data["ground_truth"]) else None
    epochs = np.arange(1, len(tr_loss) + 1) if tr_loss is not None else None

    # 1. Loss curves
    try:
        if tr_loss is not None and val_loss is not None:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.title("SPR_BENCH Loss Curves (Sequence Classification)")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        else:
            print("Loss data unavailable, skipping loss curve plot.")
    except Exception as e:
        print(f"Error creating loss curve: {e}")
    finally:
        plt.close()

    # 2. Macro-F1 curves
    try:
        if tr_macro is not None and val_macro is not None:
            plt.figure()
            plt.plot(epochs, tr_macro, label="Train Macro-F1")
            plt.plot(epochs, val_macro, label="Validation Macro-F1")
            plt.title("SPR_BENCH Macro-F1 Curves (Sequence Classification)")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "spr_bench_macroF1_curves.png"))
        else:
            print("Macro-F1 data unavailable, skipping F1 curve plot.")
    except Exception as e:
        print(f"Error creating Macro-F1 curve: {e}")
    finally:
        plt.close()

    # 3. Confusion matrix on test set
    try:
        if preds is not None and gts is not None:
            num_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title("SPR_BENCH Confusion Matrix (Test Set)")
            plt.xlabel("Predicted Class")
            plt.ylabel("Ground Truth Class")
            ticks = np.arange(num_classes)
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
            # print test accuracy
            test_acc = (preds == gts).mean()
            print(f"Test accuracy from saved predictions: {test_acc*100:.2f}%")
        else:
            print("Prediction data unavailable, skipping confusion matrix plot.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    finally:
        plt.close()
else:
    print("SPR_BENCH data not found in experiment_data.npy")
