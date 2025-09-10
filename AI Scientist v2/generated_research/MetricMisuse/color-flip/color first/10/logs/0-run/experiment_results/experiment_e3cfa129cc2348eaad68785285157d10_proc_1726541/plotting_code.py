import matplotlib.pyplot as plt
import numpy as np
import os

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to parse hidden size from key "SPR_BENCH_h{hdim}"
def get_hdim(key):
    try:
        return int(key.split("_h")[-1])
    except Exception:
        return None


hidden_dict = experiment_data.get("hidden_dim_sweep", {})
keys_sorted = sorted(hidden_dict.keys(), key=get_hdim)[:5]  # ensure at most 5

final_val_scores = []
final_hdims = []

# One plot per hidden dim
for k in keys_sorted:
    data = hidden_dict[k]
    hdim = get_hdim(k)
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    train_cwa = data["metrics"]["train_CompWA"]
    val_cwa = data["metrics"]["val_CompWA"]
    epochs = range(1, len(train_loss) + 1)

    # Save final CompWA for summary plot
    if val_cwa:
        final_val_scores.append(val_cwa[-1])
        final_hdims.append(hdim)

    # Plot losses and CompWA
    try:
        fig, ax1 = plt.subplots()
        ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue")
        ax1.plot(epochs, val_loss, label="Val Loss", color="tab:cyan")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("BCE Loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(epochs, train_cwa, label="Train CompWA", color="tab:red")
        ax2.plot(epochs, val_cwa, label="Val CompWA", color="tab:orange")
        ax2.set_ylabel("CompWA", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        plt.title(f"SPR_BENCH Hidden {hdim}: Loss & CompWA Curves")

        fname = f"SPR_BENCH_h{hdim}_loss_and_CompWA.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for hidden {hdim}: {e}")
        plt.close()

# Summary bar chart of final Val CompWA
try:
    plt.figure()
    plt.bar([str(h) for h in final_hdims], final_val_scores, color="tab:green")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Final Val CompWA")
    plt.title("SPR_BENCH: Final Validation CompWA vs Hidden Dim")
    for i, v in enumerate(final_val_scores):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_val_CompWA_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating summary bar chart: {e}")
    plt.close()

# Print best hidden dim based on final Val CompWA
if final_val_scores:
    best_idx = int(np.argmax(final_val_scores))
    print(
        f"Best hidden_dim: {final_hdims[best_idx]} with Val CompWA = {final_val_scores[best_idx]:.4f}"
    )
