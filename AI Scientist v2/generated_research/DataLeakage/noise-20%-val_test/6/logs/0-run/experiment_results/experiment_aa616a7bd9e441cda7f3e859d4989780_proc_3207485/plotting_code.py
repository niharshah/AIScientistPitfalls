import matplotlib.pyplot as plt
import numpy as np
import os

# --------- setup ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data ---------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})

# --------- containers for summary plot ---------
summary_lrs, summary_test_acc = [], []

# --------- per-run plots ---------
for run_key, run_data in runs.items():
    lr_str = run_key.split("_")[-1]  # extracts the numeric part
    epochs = np.arange(1, len(run_data["metrics"]["train_acc"]) + 1)

    # Accuracy curve
    try:
        plt.figure()
        plt.plot(epochs, run_data["metrics"]["train_acc"], label="Train")
        plt.plot(epochs, run_data["metrics"]["val_acc"], label="Validation")
        plt.title(f"SPR_BENCH Accuracy Curves (lr={lr_str})\nBlue: Train, Orange: Val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"SPR_BENCH_accuracy_lr_{lr_str}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for lr={lr_str}: {e}")
        plt.close()

    # Loss curve
    try:
        plt.figure()
        plt.plot(epochs, run_data["losses"]["train"], label="Train")
        plt.plot(epochs, run_data["losses"]["val"], label="Validation")
        plt.title(f"SPR_BENCH Loss Curves (lr={lr_str})\nBlue: Train, Orange: Val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"SPR_BENCH_loss_lr_{lr_str}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for lr={lr_str}: {e}")
        plt.close()

    # Rule fidelity curve
    try:
        plt.figure()
        plt.plot(epochs, run_data["metrics"]["rule_fidelity"], color="green")
        plt.title(f"SPR_BENCH Rule Fidelity (lr={lr_str})\nGreen: Fidelity per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Fidelity")
        fname = f"SPR_BENCH_rule_fidelity_lr_{lr_str}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot for lr={lr_str}: {e}")
        plt.close()

    # collect for summary
    summary_lrs.append(lr_str)
    summary_test_acc.append(run_data.get("test_acc", 0.0))

# --------- summary bar plot ---------
try:
    plt.figure()
    plt.bar(summary_lrs, summary_test_acc, color="skyblue")
    plt.title("SPR_BENCH Test Accuracy across Learning Rates\nBars: Test Accuracy")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    fname = "SPR_BENCH_test_accuracy_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating summary bar plot: {e}")
    plt.close()

# --------- print numeric summary ---------
for lr, acc in zip(summary_lrs, summary_test_acc):
    print(f"lr={lr}: test_acc={acc:.3f}")
