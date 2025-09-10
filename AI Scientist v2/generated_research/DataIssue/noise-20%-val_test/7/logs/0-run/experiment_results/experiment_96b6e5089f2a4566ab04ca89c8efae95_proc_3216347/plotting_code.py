import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# 0. House-keeping
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load experiment data
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    logs = experiment_data["char_unigram"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    logs = None

if logs is not None:
    train_acc = np.asarray(logs["metrics"]["train_acc"])
    val_acc = np.asarray(logs["metrics"]["val_acc"])
    train_loss = np.asarray(logs["losses"]["train"])
    val_loss = np.asarray(logs["metrics"]["val_loss"])
    val_rfs = np.asarray(logs["metrics"]["val_rfs"])
    preds = np.asarray(logs["predictions"])
    gts = np.asarray(logs["ground_truth"])
    test_acc = logs["test_acc"]
    test_rfs = logs["test_rfs"]

    # ------------------------------------------------------------------
    # 2. Plotting
    # ------------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(train_acc, label="Train Acc")
        plt.plot(val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(val_rfs, label="Val RFS")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity Score")
        plt.title("SPR_BENCH: Validation Rule Fidelity Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rfs_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RFS plot: {e}")
        plt.close()

    try:
        # Confusion matrix heat-map
        from itertools import product

        classes = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title("SPR_BENCH: Confusion Matrix (Ground Truth vs Predictions)")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.colorbar()
        plt.xticks(classes)
        plt.yticks(classes)
        # Annotate counts
        for i, j in product(range(len(classes)), repeat=2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 3. Print evaluation metrics
    # ------------------------------------------------------------------
    print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
    print(f"FINAL TEST RFS     : {test_rfs:.4f}")
else:
    print("No logs available to plot.")
