import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["dropout_rate"]["SPR_BENCH"]["runs"]
    dr_values = sorted(runs.keys(), key=float)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs, dr_values = {}, []

# Figure 1: accuracy curves ---------------------------------------------------
try:
    plt.figure()
    for dr in dr_values:
        ep = range(1, len(runs[dr]["metrics"]["train_acc"]) + 1)
        plt.plot(ep, runs[dr]["metrics"]["train_acc"], "-o", label=f"train dr={dr}")
        plt.plot(ep, runs[dr]["metrics"]["val_acc"], "--o", label=f"val dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training/Validation Accuracy - SPR_BENCH (Dropout Search)")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Figure 2: loss curves -------------------------------------------------------
try:
    plt.figure()
    for dr in dr_values:
        ep = range(1, len(runs[dr]["losses"]["train"]) + 1)
        plt.plot(ep, runs[dr]["losses"]["train"], "-o", label=f"train dr={dr}")
        plt.plot(ep, runs[dr]["losses"]["val"], "--o", label=f"val dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training/Validation Loss - SPR_BENCH (Dropout Search)")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Figure 3: URA curves --------------------------------------------------------
try:
    plt.figure()
    for dr in dr_values:
        ep = range(1, len(runs[dr]["metrics"]["val_ura"]) + 1)
        plt.plot(ep, runs[dr]["metrics"]["val_ura"], "-o", label=f"dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation URA")
    plt.title("Validation URA - SPR_BENCH (Dropout Search)")
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_URA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating URA plot: {e}")
    plt.close()

# Figure 4: final test accuracy vs dropout ------------------------------------
try:
    plt.figure()
    test_accs = [runs[dr]["test_acc"] for dr in dr_values]
    plt.plot([float(d) for d in dr_values], test_accs, "-o")
    plt.xlabel("Dropout rate")
    plt.ylabel("Test Accuracy")
    plt.title("Final Test Accuracy vs Dropout - SPR_BENCH")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy_vs_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()
