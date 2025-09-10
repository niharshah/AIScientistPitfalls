import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- Helper extraction ----------
model_key = "no_recurrent_mean_pool"
data_key = "SPR_BENCH"
runs = experiment_data.get(model_key, {}).get(data_key, {})
if not runs:
    print("No runs found in experiment_data; exiting.")
    exit()

sorted_epochs = sorted(runs.keys(), key=lambda x: int(x))
# Pick up to 3 evenly spaced runs
indices = np.linspace(
    0, len(sorted_epochs) - 1, num=min(3, len(sorted_epochs)), dtype=int
)
chosen_runs = [sorted_epochs[i] for i in indices]

# ---------- Per-run plots ----------
for ep in chosen_runs:
    try:
        r = runs[ep]
        losses = r["losses"]
        metrics = r["metrics"]
        epochs = np.arange(1, len(losses["train"]) + 1)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # left subplot: loss curves
        axs[0].plot(epochs, losses["train"], label="train")
        axs[0].plot(epochs, losses["val"], label="val")
        axs[0].set_title("Fine-tune Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Cross-entropy")
        axs[0].legend()

        # right subplot: metrics
        axs[1].plot(epochs, metrics["SWA"], label="SWA")
        axs[1].plot(epochs, metrics["CWA"], label="CWA")
        axs[1].plot(epochs, metrics["SCHM"], label="SCHM")
        axs[1].set_title("Metrics")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Score")
        axs[1].legend()

        fig.suptitle(f"SPR_BENCH | Pretrain epochs={ep}")
        fname = f"SPR_BENCH_loss_metrics_pretrain{ep}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting run {ep}: {e}")
        plt.close()

# ---------- Aggregate figure ----------
try:
    final_SWA = [runs[ep]["metrics"]["SWA"][-1] for ep in sorted_epochs]
    final_CWA = [runs[ep]["metrics"]["CWA"][-1] for ep in sorted_epochs]
    final_SCHM = [runs[ep]["metrics"]["SCHM"][-1] for ep in sorted_epochs]
    x = [int(ep) for ep in sorted_epochs]

    plt.figure(figsize=(6, 4))
    plt.plot(x, final_SWA, "o-", label="SWA")
    plt.plot(x, final_CWA, "s-", label="CWA")
    plt.plot(x, final_SCHM, "^-", label="SCHM")
    plt.title("Final Metrics vs Pre-training Epochs\nSPR_BENCH")
    plt.xlabel("Pre-training epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics_vs_pretrain.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregate plot: {e}")
    plt.close()
