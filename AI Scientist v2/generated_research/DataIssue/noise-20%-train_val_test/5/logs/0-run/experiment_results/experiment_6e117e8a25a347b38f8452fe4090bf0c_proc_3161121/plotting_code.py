import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data -------- #
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ed = experiment_data["learning_rate"]["SPR_BENCH"]
    lr_values = ed["lr_values"]  # list of floats
    epochs = ed["epochs"]  # list of ints
    train_acc = ed["metrics"]["train_acc"]  # list[list]
    val_acc = ed["metrics"]["val_acc"]  # list[list]
    test_acc = ed["test_acc"]  # list
    # ------- per-LR accuracy curves -------- #
    for i, lr in enumerate(lr_values):
        try:
            plt.figure()
            plt.plot(epochs, train_acc[i], label="Train Acc")
            plt.plot(epochs, val_acc[i], label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"SPR_BENCH Accuracy Curves (lr={lr})")
            plt.legend()
            fname = f"SPR_BENCH_lr{lr}_accuracy_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for lr={lr}: {e}")
            plt.close()

    # ------- test accuracy bar chart -------- #
    try:
        plt.figure()
        plt.bar([str(lr) for lr in lr_values], test_acc, color="skyblue")
        plt.xlabel("Learning Rate")
        plt.ylabel("Test Accuracy")
        plt.title("SPR_BENCH Test Accuracy vs Learning Rate")
        fname = "SPR_BENCH_test_accuracy_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating test-accuracy plot: {e}")
        plt.close()

    # ------- print numeric results -------- #
    print("\nTest Accuracy Table:")
    for lr, acc in zip(lr_values, test_acc):
        print(f"  lr={lr:.4g} -> test_acc={acc:.4f}")
