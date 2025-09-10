import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    runs = experiment_data.get("shape_color_split", {}).get("SPR_BENCH", {})
    # plot at most five runs
    for i, (epochs_str, run) in enumerate(
        sorted(runs.items(), key=lambda x: int(x[0]))
    ):
        if i >= 5:
            break
        try:
            # extract data
            train_loss = run["losses"]["train"]
            val_loss = run["losses"]["val"]
            hwa = run["metrics"]["val"]
            ep_range = np.arange(1, len(train_loss) + 1)

            # figure
            fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
            fig.suptitle(f"SPR_BENCH – {epochs_str} Epochs")

            # losses
            ax[0].plot(ep_range, train_loss, label="Train Loss")
            ax[0].plot(ep_range, val_loss, label="Val Loss")
            ax[0].set_ylabel("Cross-Entropy Loss")
            ax[0].legend()
            ax[0].set_title("Training vs. Validation Loss")

            # HWA
            ax[1].plot(ep_range, hwa, color="green", label="Val HWA")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("HWA")
            ax[1].legend()
            ax[1].set_title("Validation Harmonic Weighted Accuracy")

            # save
            fname = f"SPR_BENCH_loss_hwa_{epochs_str}ep.png"
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(working_dir, fname))
            plt.close(fig)

            # print final metric
            print(f"Run {epochs_str} epochs – final HWA: {hwa[-1]:.3f}")
        except Exception as e:
            print(f"Error creating plot for {epochs_str} epochs: {e}")
            plt.close()
