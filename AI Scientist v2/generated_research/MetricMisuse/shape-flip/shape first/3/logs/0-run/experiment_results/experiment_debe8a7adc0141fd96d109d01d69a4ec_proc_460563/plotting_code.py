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
    datasets = experiment_data["MSDT"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    datasets = {}


# ---------- helper for test accuracy ----------
def simple_acc(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return (np.array(y_true) == np.array(y_pred)).mean()


# ---------- 1) Train / Val loss curves ----------
try:
    plt.figure()
    epochs = list(range(1, len(datasets["SPR_BENCH"]["losses"]["train"]) + 1))
    plt.plot(epochs, datasets["SPR_BENCH"]["losses"]["train"], label="Train", lw=2)
    for name, data in datasets.items():
        vloss = data["losses"]["val"]
        plt.plot(epochs[: len(vloss)], vloss, label=f"Val {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "MSDT – Training and Validation Losses\nLeft: Train, Right: Per-Dataset Validation (SPR Bench, Token Renamed, Color Shuffled)"
    )
    plt.legend()
    fname = os.path.join(working_dir, "MSDT_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2) Validation SWA curves ----------
try:
    plt.figure()
    for name, data in datasets.items():
        vswa = data["metrics"]["val"]
        plt.plot(range(1, len(vswa) + 1), vswa, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title(
        "Validation SWA over Epochs\nLeft: Ground Truth, Right: Predicted – Dev Sets"
    )
    plt.legend()
    fname = os.path.join(working_dir, "MSDT_val_SWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------- 3) Test accuracy bar chart ----------
try:
    test_names, test_accs = [], []
    for name, data in datasets.items():
        acc = simple_acc(data["ground_truth"], data["predictions"])
        test_names.append(name)
        test_accs.append(acc)
    plt.figure()
    plt.bar(test_names, test_accs, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy per Dataset\nLeft: Ground Truth, Right: Predictions")
    for i, v in enumerate(test_accs):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fname = os.path.join(working_dir, "MSDT_test_accuracy.png")
    plt.savefig(fname)
    plt.close()
    # print numeric results
    for n, a in zip(test_names, test_accs):
        print(f"{n} Test Accuracy: {a:.4f}")
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()
