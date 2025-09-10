import matplotlib.pyplot as plt
import numpy as np
import os

# set up paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    lr_dict = experiment_data["learning_rate_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    lr_dict = {}

# ---------------------- Plot 1: loss curves ------------------------ #
try:
    plt.figure(figsize=(6, 4))
    for lr, data in lr_dict.items():
        epochs = data["metrics"]["epochs"]
        plt.plot(epochs, data["metrics"]["train_loss"], "--", label=f"train lr={lr}")
        plt.plot(epochs, data["metrics"]["val_loss"], "-", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss\n(Transformer classifier)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------------- Plot 2: accuracy curves -------------------- #
try:
    plt.figure(figsize=(6, 4))
    for lr, data in lr_dict.items():
        epochs = data["metrics"]["epochs"]
        plt.plot(epochs, data["metrics"]["train_acc"], "--", label=f"train lr={lr}")
        plt.plot(epochs, data["metrics"]["val_acc"], "-", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs Validation Accuracy\n(Transformer classifier)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ---------------------- Plot 3: test accuracy bar ------------------ #
try:
    plt.figure(figsize=(5, 4))
    lrs, test_accs = zip(*[(lr, data["test_acc"]) for lr, data in lr_dict.items()])
    x = np.arange(len(lrs))
    plt.bar(x, test_accs, color="skyblue")
    plt.xticks(x, lrs)
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH: Final Test Accuracy by Learning Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()

# ---------------------- Print evaluation metrics ------------------- #
for lr, data in lr_dict.items():
    print(f"LR {lr}: Test Accuracy = {data['test_acc']:.3f}")
