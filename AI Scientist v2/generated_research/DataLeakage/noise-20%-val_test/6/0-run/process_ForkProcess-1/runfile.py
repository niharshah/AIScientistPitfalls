import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper: confusion matrix without sklearn
def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


for ds_name, ds_dict in experiment_data.items():
    metrics = ds_dict.get("metrics", {})
    losses = ds_dict.get("losses", {})
    preds = ds_dict.get("predictions", None)
    gt = ds_dict.get("ground_truth", None)

    epochs = np.arange(1, len(losses.get("train", [])) + 1)

    # 1) Loss curves ---------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Loss Curves\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) Accuracy curves -----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, metrics["train_acc"], label="Train Acc")
        plt.plot(epochs, metrics["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Accuracy Curves\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # 3) Rule fidelity -------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, metrics["rule_fidelity"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Fidelity")
        plt.title(f"{ds_name} Rule Fidelity over Epochs")
        fname = os.path.join(working_dir, f"{ds_name}_rule_fidelity.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot for {ds_name}: {e}")
        plt.close()

    # 4) Confusion matrix ----------------------------------------------------
    try:
        if preds is not None and gt is not None:
            num_classes = len(np.unique(gt))
            cm = confusion_matrix_np(gt, preds, num_classes)
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{ds_name} Confusion Matrix\nTest Set")
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # Print final metric -----------------------------------------------------
    try:
        if preds is not None and gt is not None:
            test_acc = (preds == gt).mean()
            print(f"{ds_name} final test accuracy: {test_acc:.3f}")
    except Exception as e:
        print(f"Error computing final accuracy for {ds_name}: {e}")
