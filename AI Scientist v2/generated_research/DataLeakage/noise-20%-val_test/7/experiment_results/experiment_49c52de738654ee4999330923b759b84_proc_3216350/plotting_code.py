import matplotlib.pyplot as plt
import numpy as np
import os

# setup -----------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper ----------------------------------------------------------------
def get_metric(abl, metric):
    return experiment_data[abl]["SPR_BENCH"]["metrics"][metric]


ablations = list(experiment_data.keys())

# 1. Train / Val accuracy ------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for abl in ablations:
        epochs = np.arange(len(get_metric(abl, "train_acc")))
        plt.plot(epochs, get_metric(abl, "train_acc"), "--", label=f"{abl} Train")
        plt.plot(epochs, get_metric(abl, "val_acc"), "-", label=f"{abl} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "SPR_BENCH: Train vs Validation Accuracy\n(Left: Train, Right: Val curves)"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2. Validation RFS ------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for abl in ablations:
        epochs = np.arange(len(get_metric(abl, "val_rfs")))
        plt.plot(epochs, get_metric(abl, "val_rfs"), label=abl)
    plt.xlabel("Epoch")
    plt.ylabel("Rule‐Fidelity Score")
    plt.title("SPR_BENCH: Validation Rule-Fidelity Score (RFS)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_rfs.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating RFS plot: {e}")
    plt.close()

# 3. Training loss -------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for abl in ablations:
        epochs = np.arange(len(experiment_data[abl]["SPR_BENCH"]["losses"]["train"]))
        plt.plot(
            epochs, experiment_data[abl]["SPR_BENCH"]["losses"]["train"], label=abl
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("SPR_BENCH: Training Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_training_loss.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 4. Test performance ----------------------------------------------------
try:
    labels = ablations
    test_acc = [experiment_data[a]["SPR_BENCH"]["test_acc"] for a in labels]
    test_rfs = [experiment_data[a]["SPR_BENCH"]["test_rfs"] for a in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, test_acc, width, label="Test Acc")
    plt.bar(x + width / 2, test_rfs, width, label="Test RFS")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(
        "SPR_BENCH: Final Test Metrics\nLeft bars: Accuracy, Right bars: Rule Fidelity"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()
