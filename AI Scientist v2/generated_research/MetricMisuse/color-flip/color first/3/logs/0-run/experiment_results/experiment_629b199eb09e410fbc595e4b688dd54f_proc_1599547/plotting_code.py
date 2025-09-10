import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

# ensure working directory exists
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- #
# 1. Load experiment data                                               #
# --------------------------------------------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is None:
    exit()

ds = "SPR_BENCH"
ds_data = experiment_data[ds]

# --------------------------------------------------------------------- #
# 2. Plot train/val loss                                                #
# --------------------------------------------------------------------- #
try:
    plt.figure()
    train_epochs, train_losses = zip(*ds_data["losses"]["train"])
    val_epochs, val_losses = zip(*ds_data["losses"]["val"])
    plt.plot(train_epochs, train_losses, label="Train Loss")
    plt.plot(val_epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title(f"{ds}: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds}_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------------------------------------------------------------------- #
# 3. Plot validation metrics                                            #
# --------------------------------------------------------------------- #
try:
    plt.figure()
    epochs, cwa, swa, hcs = zip(*ds_data["metrics"]["val"])
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, hcs, label="HCSA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{ds}: Validation Metrics over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds}_val_metric_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()


# --------------------------------------------------------------------- #
# 4. Confusion matrices (dev & test)                                    #
# --------------------------------------------------------------------- #
def plot_cm(y_true, y_pred, split_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"{ds} {split_name}: Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=6,
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


for split in ["dev", "test"]:
    try:
        plt.figure(figsize=(6, 5))
        y_true = ds_data["ground_truth"][split]
        y_pred = ds_data["predictions"][split]
        plot_cm(y_true, y_pred, split.capitalize())
        fname = os.path.join(working_dir, f"{ds}_{split}_confusion_matrix.png")
        plt.savefig(fname, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating {split} confusion matrix: {e}")
        plt.close()
