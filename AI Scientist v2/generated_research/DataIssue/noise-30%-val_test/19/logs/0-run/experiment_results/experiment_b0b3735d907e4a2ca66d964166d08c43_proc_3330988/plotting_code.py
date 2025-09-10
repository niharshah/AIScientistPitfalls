import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("SPR_BENCH", {})

# ----------------- Plot 1: macro-F1 curves -----------------
try:
    per_lr = spr_data.get("per_lr", {})
    if per_lr:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for lr, d in per_lr.items():
            epochs = np.arange(1, len(d["metrics"]["train_macro_F1"]) + 1)
            axs[0].plot(epochs, d["metrics"]["train_macro_F1"], label=f"lr={lr}")
            axs[1].plot(epochs, d["metrics"]["val_macro_F1"], label=f"lr={lr}")
        for ax, split in zip(axs, ["Train", "Validation"]):
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro-F1")
            ax.set_title(f"{split} Macro-F1")
            ax.legend()
        plt.suptitle("SPR_BENCH: Macro-F1 over Epochs (Left: Train, Right: Val)")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fname = os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating macro-F1 plot: {e}")
    plt.close()

# ----------------- Plot 2: best-lr loss curves -----------------
try:
    best_lr = spr_data.get("best_lr", None)
    if best_lr is not None:
        losses = spr_data["per_lr"][best_lr]["losses"]
        epochs = np.arange(1, len(losses["train"]) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, losses["train"], label="Train")
        plt.plot(epochs, losses["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title(f"SPR_BENCH: Loss Curve (best lr={best_lr})")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_best_lr_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------- Plot 3: confusion matrix -----------------
try:
    preds = np.array(spr_data.get("predictions", []))
    gts = np.array(spr_data.get("ground_truth", []))
    if preds.size and gts.size:
        tp = np.sum((preds == 1) & (gts == 1))
        tn = np.sum((preds == 0) & (gts == 0))
        fp = np.sum((preds == 1) & (gts == 0))
        fn = np.sum((preds == 0) & (gts == 1))
        cm = np.array([[tn, fp], [fn, tp]])
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("SPR_BENCH: Test Confusion Matrix")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

print("Plotting complete. Files saved to:", working_dir)
