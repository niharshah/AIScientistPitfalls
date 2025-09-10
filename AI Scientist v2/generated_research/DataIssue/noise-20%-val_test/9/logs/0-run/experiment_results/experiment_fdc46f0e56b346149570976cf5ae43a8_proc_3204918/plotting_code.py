import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
exp_file = os.path.join(os.getcwd(), "experiment_data.npy")
try:
    experiment_data = np.load(exp_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# we only expect one dataset / run
ds_key = None
if experiment_data:
    ds_key = list(experiment_data.get("epoch_tuning", {}).keys())[0]

if ds_key:
    data = experiment_data["epoch_tuning"][ds_key]
    met = data["metrics"]
    loss = data["losses"]
    epochs = range(1, len(met["train_acc"]) + 1)

    # 1. Loss curves ----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, loss["train"], label="Train")
        plt.plot(epochs, loss["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_key} – Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. Accuracy curves ------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, met["train_acc"], label="Train")
        plt.plot(epochs, met["val_acc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_key} – Train vs Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_acc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3. RBA over epochs -----------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, data["RBA"][: len(epochs)], label="Validation RBA")
        if len(data["RBA"]) > len(epochs):
            plt.scatter(len(epochs) + 1, data["RBA"][-1], color="red", label="Test RBA")
        plt.xlabel("Epoch")
        plt.ylabel("RBA")
        plt.title(f"{ds_key} – Rule-Based Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key}_RBA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating RBA plot: {e}")
        plt.close()

    # 4. Confusion matrix ----------------------------------------------
    try:
        preds, gts = np.array(data["predictions"]), np.array(data["ground_truth"])
        if preds.size and gts.size and preds.shape == gts.shape:
            num_cls = int(max(gts.max(), preds.max()) + 1)
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{ds_key} – Confusion Matrix (Test)")
            fname = os.path.join(working_dir, f"{ds_key}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        else:
            raise ValueError("Predictions or ground_truth missing")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # 5. Class distribution histogram ----------------------------------
    try:
        if preds.size and gts.size:
            plt.figure()
            bins = np.arange(0, max(preds.max(), gts.max()) + 2) - 0.5
            plt.hist(gts, bins=bins, alpha=0.6, label="Ground Truth")
            plt.hist(preds, bins=bins, alpha=0.6, label="Predictions")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(f"{ds_key} – Class Distribution (Test)")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_key}_class_dist.png")
            plt.savefig(fname)
            plt.close()
        else:
            raise ValueError("Predictions or ground_truth missing")
    except Exception as e:
        print(f"Error creating class distribution plot: {e}")
        plt.close()

    # ---------- print metrics ----------
    print(
        f"Test Accuracy: {met['test_acc']:.3f} | "
        f"Test Loss: {loss['test']:.4f} | "
        f"Test RBA: {data['RBA'][-1]:.3f}"
    )
else:
    print("No dataset found inside experiment data.")
