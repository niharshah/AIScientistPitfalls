import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- Load data ---------------- #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ls_dict = exp.get("LABEL_SMOOTHING", {})
    ls_values = sorted(ls_dict.keys(), key=float)  # keep numeric order
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ls_dict, ls_values = {}, []

epochs = None
# ---------------- Accuracy curves ---------------- #
try:
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for sm in ls_values:
        data = ls_dict[sm]
        train_acc = data["metrics"]["train_acc"]
        val_acc = data["metrics"]["val_acc"]
        if epochs is None:
            epochs = range(1, len(train_acc) + 1)
        ax1.plot(epochs, train_acc, label=f"sm={sm}")
        ax2.plot(epochs, val_acc, label=f"sm={sm}")
    ax1.set_title("Train Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Acc")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Acc")
    plt.suptitle("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
    ax1.legend()
    ax2.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------------- Loss curves ---------------- #
try:
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for sm in ls_values:
        data = ls_dict[sm]
        train_loss = data["losses"]["train"]
        val_loss = data["losses"]["val"]
        ax1.plot(epochs, train_loss, label=f"sm={sm}")
        ax2.plot(epochs, val_loss, label=f"sm={sm}")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    plt.suptitle("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    ax1.legend()
    ax2.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------- Test accuracy bar chart ---------------- #
try:
    test_accs = [ls_dict[sm]["metrics"]["test_acc"] for sm in ls_values]
    plt.figure(figsize=(6, 4))
    plt.bar(ls_values, test_accs, color="steelblue")
    plt.ylabel("Accuracy")
    plt.xlabel("Label Smoothing")
    plt.title("SPR_BENCH Test Accuracy vs. Label Smoothing")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar chart: {e}")
    plt.close()
