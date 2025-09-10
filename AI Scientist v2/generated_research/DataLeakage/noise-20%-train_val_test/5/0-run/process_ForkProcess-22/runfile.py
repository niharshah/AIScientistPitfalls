import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    results = experiment_data["SingleTransformerLayer"]["SPR_BENCH"]["results"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    results = {}

nheads, test_accs = [], []
train_acc_curves, val_acc_curves = {}, {}
train_loss_curves, val_loss_curves = {}, {}

# Extract metrics
for nh, res in results.items():
    nheads.append(int(nh))
    test_accs.append(res["test_acc"])
    train_acc_curves[nh] = res["metrics"]["train"]
    val_acc_curves[nh] = res["metrics"]["val"]
    train_loss_curves[nh] = res["losses"]["train"]
    val_loss_curves[nh] = res["losses"]["val"]
    print(f"nhead={nh}: test_acc={res['test_acc']:.4f}")

# Plot 1: Accuracy curves
try:
    plt.figure()
    for nh in sorted(train_acc_curves, key=int):
        epochs = range(1, len(train_acc_curves[nh]) + 1)
        plt.plot(epochs, train_acc_curves[nh], label=f"{nh}-head Train")
        plt.plot(epochs, val_acc_curves[nh], linestyle="--", label=f"{nh}-head Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves - SPR_BENCH\nTrain vs Validation for each nhead")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 2: Loss curves
try:
    plt.figure()
    for nh in sorted(train_loss_curves, key=int):
        epochs = range(1, len(train_loss_curves[nh]) + 1)
        plt.plot(epochs, train_loss_curves[nh], label=f"{nh}-head Train")
        plt.plot(epochs, val_loss_curves[nh], linestyle="--", label=f"{nh}-head Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curves - SPR_BENCH\nTrain vs Validation for each nhead")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot 3: Test accuracy bar chart
try:
    plt.figure()
    plt.bar([str(nh) for nh in nheads], test_accs, color="skyblue")
    plt.xlabel("Number of Attention Heads")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)
    plt.title("Final Test Accuracy - SPR_BENCH\nSingle-Layer Transformer Ablation")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating test-accuracy bar plot: {e}")
    plt.close()
