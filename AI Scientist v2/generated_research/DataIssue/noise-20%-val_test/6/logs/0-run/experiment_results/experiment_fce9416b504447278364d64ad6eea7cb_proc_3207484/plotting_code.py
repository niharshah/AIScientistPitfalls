import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("adam_beta1", {}).get("SPR_BENCH", {})
if not spr_data:
    print("No SPR_BENCH data found, aborting plot generation.")
    exit()

betas = sorted(spr_data.keys(), key=float)

# 1) Accuracy curves
try:
    plt.figure()
    for beta in betas:
        epochs = range(1, len(spr_data[beta]["metrics"]["train_acc"]) + 1)
        plt.plot(
            epochs,
            spr_data[beta]["metrics"]["train_acc"],
            label=f"β1={beta} Train",
            linestyle="-",
        )
        plt.plot(
            epochs,
            spr_data[beta]["metrics"]["val_acc"],
            label=f"β1={beta} Val",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fname = os.path.join(working_dir, "SPR_BENCH_acc_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Loss curves
try:
    plt.figure()
    for beta in betas:
        epochs = range(1, len(spr_data[beta]["losses"]["train"]) + 1)
        plt.plot(
            epochs,
            spr_data[beta]["losses"]["train"],
            label=f"β1={beta} Train",
            linestyle="-",
        )
        plt.plot(
            epochs,
            spr_data[beta]["losses"]["val"],
            label=f"β1={beta} Val",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Rule-fidelity curves
try:
    plt.figure()
    for beta in betas:
        epochs = range(1, len(spr_data[beta]["metrics"]["rule_fidelity"]) + 1)
        plt.plot(epochs, spr_data[beta]["metrics"]["rule_fidelity"], label=f"β1={beta}")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH: Rule Fidelity vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating rule-fidelity plot: {e}")
    plt.close()

# 4) Test accuracy bar chart
try:
    plt.figure()
    test_accs = [spr_data[b]["test_acc"] for b in betas]
    plt.bar(betas, test_accs, color="skyblue")
    plt.ylabel("Accuracy")
    plt.xlabel("β1")
    plt.title("SPR_BENCH: Final Test Accuracy per β1")
    for i, acc in enumerate(test_accs):
        plt.text(i, acc + 0.002, f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar: {e}")
    plt.close()
