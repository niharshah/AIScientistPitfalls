import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- Load data ---------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

wd_dict = experiment_data["weight_decay"]["SPR_BENCH"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ---------------- Acc & Loss curves ---------------- #
try:
    plt.figure()
    for wd_key, col in zip(wd_dict.keys(), colors):
        d = wd_dict[wd_key]
        epochs = d["epochs"]
        plt.plot(epochs, d["metrics"]["train_acc"], color=col, label=f"{wd_key} train")
        plt.plot(
            epochs,
            d["metrics"]["val_acc"],
            color=col,
            linestyle="--",
            label=f"{wd_key} val",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves\nSolid: Train, Dashed: Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

try:
    plt.figure()
    for wd_key, col in zip(wd_dict.keys(), colors):
        d = wd_dict[wd_key]
        epochs = d["epochs"]
        plt.plot(epochs, d["losses"]["train"], color=col, label=f"{wd_key} train")
        plt.plot(
            epochs, d["losses"]["val"], color=col, linestyle="--", label=f"{wd_key} val"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nSolid: Train, Dashed: Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()


# ---------------- Confusion matrix for best wd ---------------- #
def accuracy(preds, gts):
    preds = np.asarray(preds)
    gts = np.asarray(gts)
    return (preds == gts).mean()


best_key, best_val_acc = None, -1
for wd_key, d in wd_dict.items():
    acc = d["metrics"]["val_acc"][-1]
    if acc > best_val_acc:
        best_val_acc, best_key = acc, wd_key

# Compute and plot confusion matrix (binary classification assumed)
try:
    best_d = wd_dict[best_key]
    preds = np.array(best_d["predictions"])
    gts = np.array(best_d["ground_truth"])
    labels = np.unique(gts)
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(labels, labels)
    plt.yticks(labels, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(
        f"SPR_BENCH Confusion Matrix (Best {best_key})\nLeft: Ground Truth, Right: Predicted"
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_confusion_{best_key}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- Print test accuracies ---------------- #
print("Final Test Accuracies:")
for wd_key, d in wd_dict.items():
    test_acc = accuracy(d["predictions"], d["ground_truth"])
    print(f"{wd_key}: {test_acc:.4f}")
