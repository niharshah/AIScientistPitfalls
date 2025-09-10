import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Iterate over every dataset stored in experiment_data
for dset, dct in experiment_data.items():
    losses = dct.get("losses", {})
    metrics = dct.get("metrics", {})
    train_l = losses.get("train", [])
    val_l = losses.get("val", [])
    train_a = metrics.get("train", [])
    val_a = metrics.get("val", [])
    val_swa = metrics.get("val_swa", [])

    # --------- Plot 1: Loss curves ---------
    try:
        plt.figure()
        if train_l:
            plt.plot(train_l, label="train")
        if val_l:
            plt.plot(val_l, label="val", linestyle="--")
        plt.title(f"{dset} Loss Curves\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dset}: {e}")
        plt.close()

    # --------- Plot 2: Accuracy curves ---------
    try:
        plt.figure()
        if train_a:
            plt.plot(train_a, label="train")
        if val_a:
            plt.plot(val_a, label="val", linestyle="--")
        plt.title(f"{dset} Accuracy Curves\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves for {dset}: {e}")
        plt.close()

    # --------- Plot 3: Shape-Weighted Accuracy (SWA) ---------
    try:
        if val_swa and any(v is not None for v in val_swa):
            plt.figure()
            plt.plot([v if v is not None else np.nan for v in val_swa], label="val SWA")
            plt.title(f"{dset} Validation Shape-Weighted Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_swa_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating SWA curve for {dset}: {e}")
        plt.close()

    # --------- Plot 4: Combined validation metrics ---------
    try:
        plt.figure()
        if val_l:
            plt.plot(val_l, label="Val Loss")
        if val_a:
            plt.plot(val_a, label="Val Acc")
        if val_swa:
            plt.plot([v if v is not None else np.nan for v in val_swa], label="Val SWA")
        plt.title(f"{dset} Validation Metrics Overview")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend(fontsize=7)
        plt.savefig(os.path.join(working_dir, f"{dset}_val_metrics_overview.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating overview plot for {dset}: {e}")
        plt.close()

    # --------- Plot 5: Final scores bar chart ---------
    try:
        final_scores, labels = [], []
        if train_a:
            labels.append("Train Acc")
            final_scores.append(train_a[-1])
        if val_a:
            labels.append("Val Acc")
            final_scores.append(val_a[-1])
        if val_swa and val_swa[-1] is not None:
            labels.append("Val SWA")
            final_scores.append(val_swa[-1])
        if final_scores:
            plt.figure()
            x = np.arange(len(final_scores))
            plt.bar(x, final_scores, color="skyblue")
            plt.xticks(x, labels)
            plt.title(f"{dset} Final Epoch Scores")
            plt.ylabel("Score")
            plt.savefig(os.path.join(working_dir, f"{dset}_final_scores.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating final score bar for {dset}: {e}")
        plt.close()
