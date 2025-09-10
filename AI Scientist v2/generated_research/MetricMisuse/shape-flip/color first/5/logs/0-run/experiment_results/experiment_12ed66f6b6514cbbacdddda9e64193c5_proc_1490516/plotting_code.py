import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["batch_size"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# Helper: collect per-batch-size arrays
batch_sizes, tr_loss, val_loss, tr_cpx, val_cpx, test_loss, test_cpx = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)
for name, rec in sorted(runs.items(), key=lambda x: int(x[0][2:])):  # bs32 -> 32
    bs = int(name[2:])
    batch_sizes.append(bs)
    tr_loss.append(rec["losses"]["train"])
    val_loss.append(rec["losses"]["val"])
    tr_cpx.append(rec["metrics"]["train"])
    val_cpx.append(rec["metrics"]["val"])
    test_loss.append(rec["losses"]["test"])
    test_cpx.append(rec["metrics"]["test"])

# Figure 1: Loss curves
try:
    plt.figure()
    for idx, bs in enumerate(batch_sizes):
        epochs = np.arange(1, len(tr_loss[idx]) + 1)
        plt.plot(epochs, tr_loss[idx], label=f"Train bs{bs}")
        plt.plot(epochs, val_loss[idx], "--", label=f"Val bs{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("SPR dataset – Training vs Validation Loss (Batch-size tuning)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves_batch_size.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Figure 2: CpxWA curves
try:
    plt.figure()
    for idx, bs in enumerate(batch_sizes):
        epochs = np.arange(1, len(tr_cpx[idx]) + 1)
        plt.plot(epochs, tr_cpx[idx], label=f"Train bs{bs}")
        plt.plot(epochs, val_cpx[idx], "--", label=f"Val bs{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-weighted accuracy")
    plt.title("SPR dataset – Training vs Validation CpxWA (Batch-size tuning)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_CpxWA_curves_batch_size.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA plot: {e}")
    plt.close()

# Figure 3: Test CpxWA bar plot
try:
    plt.figure()
    plt.bar([str(bs) for bs in batch_sizes], test_cpx, color="skyblue")
    plt.xlabel("Batch size")
    plt.ylabel("Test Complexity-weighted accuracy")
    plt.title("SPR dataset – Test CpxWA by Batch size")
    plt.savefig(os.path.join(working_dir, "SPR_test_CpxWA_barplot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test CpxWA bar plot: {e}")
    plt.close()

# Print test results
for i, bs in enumerate(batch_sizes):
    print(
        f"Batch size {bs}: test_loss={test_loss[i]:.4f}, test_CpxWA={test_cpx[i]:.4f}"
    )
