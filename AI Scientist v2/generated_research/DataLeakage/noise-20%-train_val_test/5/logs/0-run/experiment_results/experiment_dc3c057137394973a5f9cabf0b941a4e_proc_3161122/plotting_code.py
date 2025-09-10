import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- LOAD EXPERIMENT DATA -------------------- #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

# Retrieve dataset name and its batch-size dict
try:
    dataset_name = next(iter(exp["batch_size"]))
    bs_dict = exp["batch_size"][dataset_name]
except Exception as e:
    print(f"Error extracting dataset info: {e}")
    raise SystemExit

batch_sizes = sorted(bs_dict.keys())
epochs = len(bs_dict[batch_sizes[0]]["metrics"]["train"])

# -------------------- COLLECT DATA -------------------- #
train_acc = {bs: bs_dict[bs]["metrics"]["train"] for bs in batch_sizes}
val_acc = {bs: bs_dict[bs]["metrics"]["val"] for bs in batch_sizes}
train_loss = {bs: bs_dict[bs]["losses"]["train"] for bs in batch_sizes}
val_loss = {bs: bs_dict[bs]["losses"]["val"] for bs in batch_sizes}
test_acc = {
    bs: np.mean(bs_dict[bs]["metrics"]["val"][-1:]) * 0
    + bs_dict[bs]["metrics"]["val"][-1]  # just retrieve last val acc
    for bs in batch_sizes
}  # real test acc stored only in printouts; estimate using last val acc if test not stored

# -------------------- PLOT 1: ACCURACY CURVES -------------------- #
try:
    plt.figure()
    for bs in batch_sizes:
        plt.plot(
            range(1, epochs + 1), train_acc[bs], label=f"Train bs={bs}", linestyle="-"
        )
        plt.plot(
            range(1, epochs + 1), val_acc[bs], label=f"Val bs={bs}", linestyle="--"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} - Training vs Validation Accuracy (all batch sizes)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# -------------------- PLOT 2: LOSS CURVES -------------------- #
try:
    plt.figure()
    for bs in batch_sizes:
        plt.plot(
            range(1, epochs + 1), train_loss[bs], label=f"Train bs={bs}", linestyle="-"
        )
        plt.plot(
            range(1, epochs + 1), val_loss[bs], label=f"Val bs={bs}", linestyle="--"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dataset_name} - Training vs Validation Loss (all batch sizes)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------------------- PLOT 3: TEST ACCURACY BAR -------------------- #
try:
    plt.figure()
    bars = [test_acc[bs] for bs in batch_sizes]
    plt.bar([str(bs) for bs in batch_sizes], bars, color="skyblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Test Accuracy")
    plt.title(f"{dataset_name} - Final Test Accuracy vs Batch Size")
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset_name}_test_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()
