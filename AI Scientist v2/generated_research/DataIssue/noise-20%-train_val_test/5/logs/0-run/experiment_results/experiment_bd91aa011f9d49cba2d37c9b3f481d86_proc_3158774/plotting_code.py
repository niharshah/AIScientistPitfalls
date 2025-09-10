import matplotlib.pyplot as plt
import numpy as np
import os

# prepare paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# retrieve SPR_BENCH results if present
spr_results = experiment_data.get("batch_size_tuning", {}).get("SPR_BENCH", {})

# iterate over batch sizes and plot
for bs, run_record in spr_results.items():
    try:
        epochs = run_record.get("epochs", [])
        m = run_record.get("metrics", {})
        tl, vl = m.get("train_loss", []), m.get("val_loss", [])
        ta, va = m.get("train_acc", []), m.get("val_acc", [])

        plt.figure(figsize=(10, 4))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tl, label="Train Loss")
        plt.plot(epochs, vl, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, ta, label="Train Acc")
        plt.plot(epochs, va, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curves")

        # Figure title and save
        plt.suptitle(f"SPR_BENCH | Batch Size {bs}\nLeft: Loss, Right: Accuracy")
        fname = f"SPR_BENCH_bs{bs}_train_val_curves.png"
        plt.savefig(os.path.join(working_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating plot for batch size {bs}: {e}")
        plt.close()

print("Finished generating plots.")
