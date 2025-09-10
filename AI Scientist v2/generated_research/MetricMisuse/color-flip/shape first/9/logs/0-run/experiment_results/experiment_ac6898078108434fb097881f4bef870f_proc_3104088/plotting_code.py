import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load ---------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_scores = {}  # collect best metric per dataset

# -------- per-dataset plots --------
for dset, data in experiment_data.items():
    # ----- extract common fields -----
    train_loss = data.get("losses", {}).get("train", [])
    val_loss = data.get("losses", {}).get("val", [])
    val_metric = data.get("metrics", {}).get("val", [])
    preds = np.array(data.get("predictions", []))
    gtruth = np.array(data.get("ground_truth", []))

    # Accuracy for quick print
    acc = float((preds == gtruth).mean()) if preds.size else float("nan")
    best_cwa = max(val_metric) if val_metric else float("nan")
    best_scores[dset] = best_cwa
    print(f"{dset}: accuracy={acc:.4f}, best_val_metric={best_cwa:.4f}")

    # ----- 1. loss curves -----
    try:
        plt.figure()
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="train")
        plt.plot(epochs, val_loss, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset} – Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {dset}: {e}")
        plt.close()

    # ----- 2. metric curves -----
    try:
        if val_metric:
            plt.figure()
            plt.plot(np.arange(1, len(val_metric) + 1), val_metric, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("CompWA")
            plt.title(f"{dset} – Validation CompWA over Epochs")
            plt.savefig(os.path.join(working_dir, f"{dset}_metric_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating metric curve plot for {dset}: {e}")
        plt.close()

    # ----- 3. confusion matrix -----
    try:
        if preds.size and gtruth.size:
            num_cls = int(max(preds.max(), gtruth.max()) + 1)
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gtruth, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.ylabel("Ground Truth")
            plt.xlabel("Predicted")
            plt.title(f"{dset} – Confusion Matrix\n(rows=GT, cols=Pred)")
            plt.savefig(os.path.join(working_dir, f"{dset}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

# -------- comparison plot across datasets --------
try:
    if best_scores:
        plt.figure()
        names = list(best_scores.keys())
        scores = [best_scores[n] for n in names]
        plt.bar(names, scores, color="skyblue")
        plt.ylabel("Best Validation CompWA")
        plt.title("Best CompWA Across Datasets")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "datasets_best_compwa.png"))
        plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()
