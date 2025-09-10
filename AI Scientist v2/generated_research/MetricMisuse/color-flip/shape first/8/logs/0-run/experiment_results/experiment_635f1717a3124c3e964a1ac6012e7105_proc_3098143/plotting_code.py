import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load data ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = list(experiment_data.keys())[:5]  # plot at most 5 experiments
final_mwa = {}

# ------------ plot per-experiment curves ------------
for tag in tags:
    try:
        data = experiment_data[tag]
        tr_loss = data["losses"]["train"]
        val_loss = data["losses"]["val"]
        val_mwa = data["metrics"]["val_MWA"]
        epochs = range(1, len(val_loss) + 1)

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, tr_loss, label="Train Loss", color="blue")
        ax1.plot(epochs, val_loss, label="Val Loss", color="orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_mwa, label="Val MWA", color="green")
        ax2.set_ylabel("MWA")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
        plt.title(f"{tag} - Loss & MWA Curves")
        save_path = os.path.join(working_dir, f"{tag}_loss_mwa_curves.png")
        plt.savefig(save_path)
        plt.close()

        if val_mwa:  # store final mwa
            final_mwa[tag] = val_mwa[-1] if val_mwa[-1] is not None else np.nan
    except Exception as e:
        print(f"Error creating plot for {tag}: {e}")
        plt.close()

# ------------ bar plot of final MWA ------------
try:
    if final_mwa:
        tags_sorted = list(final_mwa.keys())
        values = [final_mwa[t] for t in tags_sorted]
        plt.figure()
        plt.bar(tags_sorted, values, color="teal")
        plt.ylabel("Final Validation MWA")
        plt.title("Final MWA Comparison Across Experiments")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_path = os.path.join(working_dir, "final_mwa_comparison.png")
        plt.savefig(save_path)
        plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ------------ scatter for best experiment ------------
try:
    if final_mwa:
        best_tag = max(final_mwa, key=final_mwa.get)
        best_data = experiment_data[best_tag]
        preds = best_data["predictions"]
        gts = best_data["ground_truth"]
        plt.figure()
        plt.scatter(gts, preds, alpha=0.6)
        plt.xlabel("Ground Truth Labels")
        plt.ylabel("Predicted Labels")
        plt.title(f"Ground Truth vs Predicted - {best_tag}")
        plt.tight_layout()
        save_path = os.path.join(working_dir, f"{best_tag}_gt_vs_pred_scatter.png")
        plt.savefig(save_path)
        plt.close()
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    plt.close()

# ------------ print numeric results ------------
for k, v in final_mwa.items():
    print(f"{k}: final MWA={v:.4f}")
