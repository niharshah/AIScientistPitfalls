import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

# Assume one dataset key; fall back safely
ds_name = next(iter(exp["epochs"])) if "epochs" in exp else next(iter(exp))
data = exp["epochs"][ds_name]

epochs = np.arange(1, len(data["losses"]["train"]) + 1)

# 1) Loss curves --------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, data["losses"]["train"], label="Train")
    plt.plot(epochs, data["losses"]["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_name} Loss Curves\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Accuracy curves ----------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, data["metrics"]["train_acc"], label="Train")
    plt.plot(epochs, data["metrics"]["val_acc"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{ds_name} Accuracy Curves\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 3) Rule fidelity ------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, data["metrics"]["rule_fidelity"])
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    plt.title(f"{ds_name} Rule Fidelity Across Epochs")
    fname = os.path.join(working_dir, f"{ds_name}_rule_fidelity.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# 4) Confusion matrix ---------------------------------------------------------
try:
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    if preds.size and gts.size:
        num_cls = max(preds.max(), gts.max()) + 1
        cm, _, _ = np.histogram2d(
            gts, preds, bins=(num_cls, num_cls), range=[[0, num_cls], [0, num_cls]]
        )
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{ds_name} Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()

        test_acc = (preds == gts).mean()
        print(f"Test accuracy: {test_acc:.3f}")
    else:
        print("Predictions or ground truth missing; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
