import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

batch_dict = experiment_data.get("batch_size", {})
batch_sizes = sorted(int(k) for k in batch_dict.keys())

test_phas = []

for bs in batch_sizes:
    data = batch_dict[str(bs)]
    epochs = data["epochs"]
    train_loss = data["losses"]["train"]
    dev_loss = data["losses"]["dev"]
    train_pha = data["metrics"]["train_PHA"]
    dev_pha = data["metrics"]["dev_PHA"]
    test_pha = data["test_metrics"]["PHA"]
    test_phas.append((bs, test_pha))

    # --------------------------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Left subplot: Loss curves
        axes[0].plot(epochs, train_loss, label="Train Loss")
        axes[0].plot(epochs, dev_loss, label="Dev Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()

        # Right subplot: PHA curves
        axes[1].plot(epochs, train_pha, label="Train PHA")
        axes[1].plot(epochs, dev_pha, label="Dev PHA")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("PHA")
        axes[1].set_title("PHA")
        axes[1].legend()

        fig.suptitle(
            f"Batch Size={bs} | Left: Loss, Right: PHA (Synthetic/SPR Dataset)"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"bs{bs}_loss_pha_syntheticSPR.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot for batch size {bs}: {e}")
        plt.close()

# ------------------------------------------------------------
# Summary bar chart of final test PHA
try:
    if test_phas:
        bs_vals, pha_vals = zip(*sorted(test_phas))
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(bs_vals)), pha_vals, tick_label=bs_vals, color="skyblue")
        plt.ylabel("Test PHA")
        plt.xlabel("Batch Size")
        plt.title("Final Test PHA across Batch Sizes (Synthetic/SPR Dataset)")
        plt.tight_layout()
        fname = "summary_test_PHA_across_batch_sizes.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        # print numeric summary
        for bs, pha in sorted(test_phas):
            print(f"Batch Size {bs}: Test PHA = {pha:.4f}")
except Exception as e:
    print(f"Error creating summary bar chart: {e}")
    plt.close()
