import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
ds = experiment_data.get(ds_name, {})

loss_train = ds.get("losses", {}).get("train", [])
loss_val = ds.get("losses", {}).get("val", [])
bwa_train = ds.get("metrics", {}).get("train_BWA", [])
bwa_val = ds.get("metrics", {}).get("val_BWA", [])
preds = np.array(ds.get("predictions", []))
gts = np.array(ds.get("ground_truth", []))
epochs = np.arange(1, len(loss_train) + 1)

# ------------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
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

# ------------------------------------------------------------------
# 2) BWA curves
try:
    plt.figure()
    plt.plot(epochs, bwa_train, label="Train BWA")
    plt.plot(epochs, bwa_val, label="Validation BWA")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Weighted Accuracy")
    plt.title(f"{ds_name} BWA Curves\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_BWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating BWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Confusion matrix heat-map (if predictions exist)
try:
    if preds.size and gts.size:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(
            f"{ds_name} Confusion Matrix\nLeft: Ground Truth (rows), Right: Predictions (cols)"
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4) Test accuracy bar chart
try:
    if preds.size and gts.size:
        acc = (preds == gts).mean()
        plt.figure()
        plt.bar(["Accuracy"], [acc], color="green")
        plt.ylim(0, 1)
        plt.title(f"{ds_name} Test Accuracy\nPlain Accuracy over Test Set")
        for i, v in enumerate([acc]):
            plt.text(i, v + 0.02, f"{v:.2%}", ha="center")
        fname = os.path.join(working_dir, f"{ds_name}_test_accuracy.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print key metrics
if bwa_val:
    print(f"Final Validation BWA: {bwa_val[-1]:.4f}")
if preds.size and gts.size:
    print(f"Test Accuracy: {(preds == gts).mean():.4f}")
