import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# 0. House-keeping
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load experiment data
# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["no_hidden_linear"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    train_acc = np.array(exp["metrics"]["train_acc"])
    val_acc = np.array(exp["metrics"]["val_acc"])
    val_rfs = np.array(exp["metrics"]["val_rfs"])
    train_loss = np.array(exp["losses"]["train"])
    val_loss = np.array(exp["metrics"]["val_loss"])
    test_acc = exp["test_acc"]
    test_rfs = exp["test_rfs"]

    # helper to maybe subsample epochs for plotting
    def epoch_idx(arr, max_points=50):
        if len(arr) <= max_points:
            return np.arange(len(arr))
        step = max(1, len(arr) // max_points)
        return np.arange(0, len(arr), step)

    idx = epoch_idx(train_acc)

    # ------------------------------------------------------------------
    # 2. Training vs Validation Accuracy
    # ------------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(idx, train_acc[idx], label="Train Acc")
        plt.plot(idx, val_acc[idx], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves\nDataset: SPR_BENCH  |  Model: No-Hidden Linear")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_no_hidden_linear_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 3. Training vs Validation Loss
    # ------------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(idx, train_loss[idx], label="Train Loss")
        plt.plot(idx, val_loss[idx], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss Curves\nDataset: SPR_BENCH  |  Model: No-Hidden Linear")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_no_hidden_linear_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 4. Validation Rule-Fidelity Score
    # ------------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(idx, val_rfs[idx], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.ylim(0, 1.05)
        plt.title("Validation Rule-Fidelity per Epoch\nDataset: SPR_BENCH")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_no_hidden_linear_val_rfs.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating RFS plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 5. Final Test Metrics Bar Plot
    # ------------------------------------------------------------------
    try:
        plt.figure()
        metrics = ["Test Accuracy", "Test Rule-Fidelity"]
        values = [test_acc, test_rfs]
        plt.bar(metrics, values, color=["tab:blue", "tab:orange"])
        plt.ylim(0, 1.05)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.title("Final Test Metrics\nDataset: SPR_BENCH  |  Model: No-Hidden Linear")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_no_hidden_linear_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 6. Print numerical metrics
    # ------------------------------------------------------------------
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Rule-Fidelity: {test_rfs:.4f}")
