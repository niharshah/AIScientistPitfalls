import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------
# Load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Determine dataset name for file naming
dataset_name = "SPR_BENCH" if experiment_data.get("have_real", False) else "Synthetic"

# Collect final test SCWA for summary plot
lr_tags, test_scores = [], []

# ---------------------------------------------------
# Per-LR training/validation curves
for tag, res in experiment_data.get("learning_rate", {}).items():
    epochs = res.get("epochs", [])
    train_loss = res.get("losses", {}).get("train", [])
    val_loss = res.get("losses", {}).get("val", [])
    val_scwa = res.get("metrics", {}).get("val", [])
    test_scwa = res.get("test_scwa", None)

    lr_tags.append(tag)
    test_scores.append(test_scwa)

    try:
        plt.figure(figsize=(10, 4))

        # Left subplot: loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.title("Loss Curves")
        plt.legend()

        # Right subplot: validation SCWA
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_scwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("SCWA")
        plt.title("Validation SCWA")

        plt.suptitle(f"{dataset_name}: {tag}")
        fname = f"{dataset_name}_{tag}_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {tag}: {e}")
        plt.close()

# ---------------------------------------------------
# Summary bar chart of final test SCWA
try:
    plt.figure()
    plt.bar(range(len(test_scores)), test_scores, tick_label=lr_tags)
    plt.ylabel("Test SCWA")
    plt.title(f"{dataset_name}: Test SCWA by Learning Rate")
    plt.savefig(os.path.join(working_dir, f"{dataset_name}_test_scwa_summary.png"))
    plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()

# ---------------------------------------------------
# Print final metrics
print("\nFinal Test SCWA Scores")
for tag, score in zip(lr_tags, test_scores):
    print(f"{tag:10s}: {score:.4f}")
