import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch nested dict safely
def get_nested(dct, *keys, default=None):
    for k in keys:
        dct = dct.get(k, {})
    return dct or default


model_name = "FullyConnected"
dataset_name = "SPR_BENCH"

loss_train = get_nested(
    experiment_data, model_name, dataset_name, "losses", "train", default=[]
)
loss_val = get_nested(
    experiment_data, model_name, dataset_name, "losses", "val", default=[]
)
acc_train = get_nested(
    experiment_data, model_name, dataset_name, "metrics", "train", default=[]
)
acc_val = get_nested(
    experiment_data, model_name, dataset_name, "metrics", "val", default=[]
)
preds = get_nested(experiment_data, model_name, dataset_name, "predictions", default=[])
gts = get_nested(experiment_data, model_name, dataset_name, "ground_truth", default=[])

best_val_acc = max(acc_val) if acc_val else None
if best_val_acc is not None:
    print(f"Best validation accuracy: {best_val_acc:.4f}")

# ---------- plot 1: loss curve ----------
try:
    plt.figure()
    epochs = np.arange(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="train")
    plt.plot(epochs, loss_val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Loss Curve – {dataset_name} (FullyConnected)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- plot 2: accuracy curve ----------
try:
    plt.figure()
    epochs = np.arange(1, len(acc_train) + 1)
    plt.plot(epochs, acc_train, label="train")
    plt.plot(epochs, acc_val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve – {dataset_name} (FullyConnected)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{dataset_name}_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        labels = sorted(list(set(gts) | set(preds)))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"Confusion Matrix – {dataset_name}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_confusion_matrix.png"))
        plt.close()
    else:
        print("Predictions or ground_truth missing; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
