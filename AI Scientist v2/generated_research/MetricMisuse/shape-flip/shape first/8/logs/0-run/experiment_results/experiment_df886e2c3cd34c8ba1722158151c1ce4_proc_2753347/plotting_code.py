import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- data loading ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

lr_values = ed["lr_values"]  # list of lrs
train_acc = ed["metrics"]["train_acc"]  # list[n_lr][n_epoch]
val_acc = ed["metrics"]["val_acc"]
val_ura = ed["metrics"]["val_ura"]
train_loss = ed["losses"]["train"]
test_acc = ed["metrics"]["test_acc"]  # list[n_lr]
test_ura = ed["metrics"]["test_ura"]  # list[n_lr]
epochs = np.arange(1, len(train_acc[0]) + 1)


# ---------------- plotting helpers ----------------
def save_fig(fig_name):
    plt.savefig(os.path.join(working_dir, fig_name), bbox_inches="tight")
    plt.close()


# 1) accuracy curves ----------------------------------------------------------
try:
    plt.figure()
    for i, lr in enumerate(lr_values):
        plt.plot(epochs, train_acc[i], label=f"train lr={lr:.0e}")
        plt.plot(epochs, val_acc[i], "--", label=f"val lr={lr:.0e}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "SPR_BENCH: Training vs Validation Accuracy\nLeft: Train, Right (dashed): Val"
    )
    plt.legend()
    save_fig("SPR_BENCH_accuracy_curves.png")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) training loss curves ------------------------------------------------------
try:
    plt.figure()
    for i, lr in enumerate(lr_values):
        plt.plot(epochs, train_loss[i], label=f"lr={lr:.0e}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training Loss over Epochs")
    plt.legend()
    save_fig("SPR_BENCH_training_loss_curves.png")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) validation URA curves -----------------------------------------------------
try:
    plt.figure()
    for i, lr in enumerate(lr_values):
        plt.plot(epochs, val_ura[i], label=f"lr={lr:.0e}")
    plt.xlabel("Epoch")
    plt.ylabel("Unseen Rule Accuracy (URA)")
    plt.title("SPR_BENCH: Validation URA over Epochs")
    plt.legend()
    save_fig("SPR_BENCH_validation_URA_curves.png")
except Exception as e:
    print(f"Error creating URA plot: {e}")
    plt.close()

# 4) test accuracy bar chart ---------------------------------------------------
try:
    plt.figure()
    x = np.arange(len(lr_values))
    bars = plt.bar(x, test_acc, color="skyblue")
    plt.xticks(x, [f"{lr:.0e}" for lr in lr_values])
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH: Test Accuracy per Learning-Rate\nURA shown above each bar")
    for bar, ura in zip(bars, test_ura):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"URA={ura:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_fig("SPR_BENCH_test_accuracy_per_lr.png")
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()

# ---------------- console summary ----------------
print("\n=== SPR_BENCH Test Metrics ===")
for lr, acc, ura in zip(lr_values, test_acc, test_ura):
    print(f"lr={lr:.0e} | test_acc={acc:.3f} | test_URA={ura:.3f}")
