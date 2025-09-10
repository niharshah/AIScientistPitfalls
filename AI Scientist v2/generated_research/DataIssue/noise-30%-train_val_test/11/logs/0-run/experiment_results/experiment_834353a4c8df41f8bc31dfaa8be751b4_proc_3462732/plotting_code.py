import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    epochs = data["epochs"]
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    train_f1 = data["metrics"]["train_macro_f1"]
    val_f1 = data["metrics"]["val_macro_f1"]
    test_f1s = data["metrics"]["test_macro_f1"]  # len==3 for 5,10,15 epochs

    # Plot 1: Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # Plot 2: Macro-F1 curves
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curves\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves: {e}")
        plt.close()

    # Plot 3: Test Macro-F1 by epoch schedule
    try:
        plt.figure()
        schedules = ["5 epochs", "10 epochs", "15 epochs"]
        plt.bar(schedules, test_f1s, color=["skyblue", "orange", "green"])
        plt.ylim(0, 1)
        for i, v in enumerate(test_f1s):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Test Performance\nComparison Across Training Durations")
        fname = os.path.join(working_dir, "SPR_BENCH_test_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test bar plot: {e}")
        plt.close()

    # Print evaluation metrics
    print(
        "Test Macro-F1 scores:", dict(zip(["5e", "10e", "15e"], np.round(test_f1s, 4)))
    )
else:
    print("SPR_BENCH results not found in experiment_data.")
