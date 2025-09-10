import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- paths -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------- plotting -----------------
for key in experiment_data:  # e.g. 'dropout_prob_0.1'
    try:
        data = experiment_data[key]["SPR_BENCH"]
        train_loss = data["losses"]["train"]
        val_loss = data["losses"]["val"]
        comp_wa = data["metrics"]["val_CompWA"]
        epochs = np.arange(1, len(train_loss) + 1)

        plt.figure(figsize=(10, 4))

        # Left subplot: Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, "o-", label="Train Loss")
        plt.plot(epochs, val_loss, "s-", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Loss")
        plt.legend()

        # Right subplot: CompWA curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs, comp_wa, "d-", color="green", label="Val CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.ylim(0, 1)
        plt.title("Complexity Weighted Accuracy")
        plt.legend()

        plt.suptitle(
            f"SPR_BENCH – {key}\nLeft: Loss Curves, Right: Complexity Weighted Accuracy"
        )
        fname = f"SPR_BENCH_{key}_loss_compwa.png".replace(" ", "")
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {key}: {e}")
        plt.close()
