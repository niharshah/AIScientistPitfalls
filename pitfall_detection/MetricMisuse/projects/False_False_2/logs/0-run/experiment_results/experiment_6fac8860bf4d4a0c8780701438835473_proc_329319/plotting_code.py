import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("optimizer_type", {}).get("spr_bench", {})
optimizers = [
    k
    for k in spr_data.keys()
    if k not in ("best_optimizer", "predictions", "ground_truth", "test_metrics")
]

epochs = None
if optimizers:
    epochs = len(spr_data[optimizers[0]]["losses"]["train"])
    xs = np.arange(1, epochs + 1)

# ------------------------------------------------------------------
# Plot 1: Loss curves
try:
    if optimizers:
        plt.figure()
        for opt in optimizers:
            plt.plot(xs, spr_data[opt]["losses"]["train"], label=f"{opt}-train")
            plt.plot(xs, spr_data[opt]["losses"]["dev"], label=f"{opt}-dev", ls="--")
        plt.title("spr_bench: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 2: PHA curves
try:
    if optimizers:
        plt.figure()
        for opt in optimizers:
            plt.plot(xs, spr_data[opt]["metrics"]["train_PHA"], label=f"{opt}-train")
            plt.plot(
                xs, spr_data[opt]["metrics"]["dev_PHA"], label=f"{opt}-dev", ls="--"
            )
        plt.title("spr_bench: Training vs Validation PHA")
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_PHA_curves.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating PHA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 3: Test metrics for best optimizer
try:
    if "test_metrics" in spr_data:
        metrics_dict = spr_data["test_metrics"]
        names, vals = zip(*metrics_dict.items())
        plt.figure()
        plt.bar(names, vals, color=["steelblue", "orange", "green"])
        plt.ylim(0, 1)
        best_opt = spr_data.get("best_optimizer", "unknown")
        plt.title(f"spr_bench: Test Metrics (Best Optimizer: {best_opt})")
        fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test-metric plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print final test metrics
if "test_metrics" in spr_data:
    print("Test metrics:", spr_data["test_metrics"])
