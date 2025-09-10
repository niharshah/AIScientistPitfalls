import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    lrs = list(experiment_data["learning_rate"].keys())[:5]  # safety cap
    # --------- per-LR curves (≤4 figures) -------------
    for lr in lrs:
        try:
            d = experiment_data["learning_rate"][lr]
            train_loss = d["losses"]["train"]
            val_loss = d["losses"]["val"]
            val_cwa = d["metrics"]["val"]  # color-weighted accuracy
            epochs = range(1, len(train_loss) + 1)

            plt.figure(figsize=(10, 4))
            # Left subplot: losses
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Left: Train vs Val Loss")

            # Right subplot: validation metric
            plt.subplot(1, 2, 2)
            plt.plot(epochs, val_cwa, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("CWA")
            plt.title("Right: Val Color-Weighted-Accuracy")

            plt.suptitle(f"Loss & Metric Curves (Dataset: SPR) — LR={lr}")
            fname = f'lr_{lr.replace(".","p")}_loss_metric_curves.png'
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating plot for LR={lr}: {e}")
            plt.close()

    # --------- aggregated AIS comparison (1 figure) ---------
    try:
        plt.figure()
        best_ais = [
            max(experiment_data["learning_rate"][lr]["AIS"]["val"]) for lr in lrs
        ]
        plt.bar(range(len(lrs)), best_ais, tick_label=[str(lr) for lr in lrs])
        plt.xlabel("Learning Rate")
        plt.ylabel("Best AIS on Dev")
        plt.title("Aggregated AIS Performance across Learning Rates\nDataset: SPR")
        plt.savefig(os.path.join(working_dir, "aggregated_AIS_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated AIS plot: {e}")
        plt.close()
