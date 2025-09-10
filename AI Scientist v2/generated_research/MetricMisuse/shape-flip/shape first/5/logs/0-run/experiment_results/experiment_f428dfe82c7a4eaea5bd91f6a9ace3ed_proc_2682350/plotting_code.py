import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

exp_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Navigate to the SPR_BENCH record (adjust keys if you changed names)
rec = experiment_data.get("NoColorEmbedding", {}).get("SPR_BENCH", {})

losses = rec.get("losses", {})
metrics = rec.get("metrics", {})
preds = np.asarray(rec.get("predictions", []))
gts = np.asarray(rec.get("ground_truth", []))
timestamps = rec.get("timestamps", [])

epochs = np.arange(1, len(losses.get("train", [])) + 1)

# 1) Loss curve -----------------------------------------------------------
try:
    tr_loss = np.asarray(losses.get("train", []), dtype=float)
    val_loss = np.asarray(losses.get("val", []), dtype=float)
    if tr_loss.size and val_loss.size:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Loss Curves")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
    else:
        print("Loss arrays empty, skipping loss plot.")
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    plt.close()

# 2) SWA curve -----------------------------------------------------------
try:
    tr_swa = np.asarray(metrics.get("train_swa", []), dtype=float)
    val_swa = np.asarray(metrics.get("val_swa", []), dtype=float)
    if tr_swa.size and val_swa.size:
        plt.figure()
        plt.plot(epochs, tr_swa, label="Train SWA")
        plt.plot(epochs, val_swa, label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: SWA Curves")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_swa_curve.png")
        plt.savefig(fname)
    else:
        print("SWA arrays empty, skipping SWA plot.")
except Exception as e:
    print(f"Error creating SWA curve: {e}")
finally:
    plt.close()

# 3) Confusion matrix -----------------------------------------------------
try:
    if preds.size and gts.size:
        labels = np.unique(np.concatenate([preds, gts]))
        cm = np.zeros((labels.size, labels.size), dtype=int)
        for t, p in zip(gts, preds):
            cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
        plt.figure(figsize=(5, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(labels.size), labels, rotation=45)
        plt.yticks(range(labels.size), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        # add counts
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    else:
        print("Prediction/GT arrays empty, skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
finally:
    plt.close()

# Print final metrics if available
if preds.size and gts.size:
    final_loss = losses.get("val", [])[-1] if losses.get("val") else None
    final_swa = metrics.get("val_swa", [])[-1] if metrics.get("val_swa") else None
    print(f"Final Val Loss: {final_loss:.4f}" if final_loss is not None else "")
    print(f"Final Val SWA : {final_swa:.4f}" if final_swa is not None else "")
