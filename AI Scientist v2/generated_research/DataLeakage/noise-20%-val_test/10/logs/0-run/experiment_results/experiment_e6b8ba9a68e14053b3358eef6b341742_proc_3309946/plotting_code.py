import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

if experiment_data:
    dataset = list(experiment_data.keys())[0]  # 'SPR_BENCH'
    d = experiment_data[dataset]
    losses_tr = d["losses"]["train"]
    losses_val = d["losses"]["val"]
    f1_tr = d["metrics"]["train_f1"]
    f1_val = d["metrics"]["val_f1"]
    ia_val = d["metrics"]["val_interpretable_acc"]
    preds = np.array(d["predictions"])
    gts = np.array(d["ground_truth"])
    test_f1 = d["metrics"].get("test_f1", None)  # might not exist
    test_ia = d["metrics"].get("test_interpretable_acc", None)

    epochs = np.arange(1, len(losses_tr) + 1)

    # ---------------- Loss Curves ---------------- #
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset} Loss Curves\nLeft axis: Train vs Validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dataset}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------------- F1 Curves ---------------- #
    try:
        plt.figure()
        plt.plot(epochs, f1_tr, label="Train")
        plt.plot(epochs, f1_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dataset} F1 Curves\nLeft axis: Train vs Validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dataset}_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # ---------------- Interpretable Accuracy ---------------- #
    try:
        plt.figure()
        plt.plot(epochs, ia_val, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Interpretable Accuracy")
        plt.title(f"{dataset} Interpretable Accuracy on Validation Set")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dataset}_interp_acc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating interpretable accuracy plot: {e}")
        plt.close()

    # ---------------- Confusion Matrix ---------------- #
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{dataset} Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dataset}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------------- Print final metrics ---------------- #
    if test_f1 is not None and test_ia is not None:
        print(f"Final Test F1: {test_f1:.4f} | Test Interpretable Acc: {test_ia:.4f}")
