import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ Load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {}).get("runs", {})
if not runs:
    print("No runs found in experiment_data, nothing to plot.")

# Helper for colors/markers
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

# ------------------ Plot 1: loss curves ------------------
try:
    plt.figure(figsize=(7, 5))
    for i, (run, data) in enumerate(sorted(runs.items())):
        train_loss = data["losses"]["train"]
        val_loss = data["losses"]["val"]
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(
            epochs,
            train_loss,
            label=f"{run} train",
            color=colors[i % len(colors)],
            linestyle="-",
        )
        plt.plot(
            epochs,
            val_loss,
            label=f"{run} val",
            color=colors[i % len(colors)],
            linestyle="--",
        )
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ------------------ Plot 2: validation accuracy curves ------------------
try:
    plt.figure(figsize=(7, 5))
    for i, (run, data) in enumerate(sorted(runs.items())):
        val_metrics = data["metrics"]["val"]
        accs = [m["acc"] for m in val_metrics]
        epochs = np.arange(1, len(accs) + 1)
        plt.plot(epochs, accs, label=f"{run}", color=colors[i % len(colors)])
    plt.title("SPR_BENCH: Validation Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# ------------------ Plot 3: test metrics bar chart ------------------
try:
    metrics = ["acc", "cwa", "swa", "compwa"]
    lr_labels, accs, cwas, swas, compwas = [], [], [], [], []
    for run, data in sorted(runs.items()):
        test_m = data["metrics"]["test"]
        lr_labels.append(run)
        accs.append(test_m["acc"])
        cwas.append(test_m["cwa"])
        swas.append(test_m["swa"])
        compwas.append(test_m["compwa"])

    x = np.arange(len(lr_labels))
    width = 0.2
    plt.figure(figsize=(10, 5))
    plt.bar(x - 1.5 * width, accs, width, label="ACC")
    plt.bar(x - 0.5 * width, cwas, width, label="CWA")
    plt.bar(x + 0.5 * width, swas, width, label="SWA")
    plt.bar(x + 1.5 * width, compwas, width, label="CompWA")
    plt.title("SPR_BENCH: Test Metrics by Learning Rate")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(x, lr_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# ------------------ Print test metric table ------------------
header = f"{'Run':15s} |  ACC   CWA   SWA  CompWA"
print(header)
print("-" * len(header))
for run, data in sorted(runs.items()):
    t = data["metrics"]["test"]
    print(f"{run:15s} | {t['acc']:.3f} {t['cwa']:.3f} {t['swa']:.3f} {t['compwa']:.3f}")
