import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["no_view_aug"]["SPR_transformer"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

losses, metrics = run["losses"], run["metrics"]

# ------------------------------------------------------------------
# 1) Pre-training loss curve
try:
    plt.figure()
    plt.plot(range(1, len(losses["pretrain"]) + 1), losses["pretrain"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title("SPR Dataset – Contrastive Pre-training Loss")
    fname = os.path.join(working_dir, "SPR_pretrain_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating pretrain loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Fine-tuning train / val loss curves
try:
    plt.figure()
    epochs = range(1, len(losses["train"]) + 1)
    plt.plot(epochs, losses["train"], label="Train CE Loss", marker="o")
    plt.plot(epochs, losses["val"], label="Val CE Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Dataset – Fine-tuning Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_finetune_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating fine-tuning loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Validation metric curves
try:
    plt.figure()
    epochs = range(1, len(metrics["val_SCWA"]) + 1)
    plt.plot(epochs, metrics["val_SWA"], label="SWA", marker="o")
    plt.plot(epochs, metrics["val_CWA"], label="CWA", marker="s")
    plt.plot(epochs, metrics["val_SCWA"], label="SCWA", marker="^")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR Dataset – Validation Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_validation_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print best epoch summary
val_scwa = np.array(metrics["val_SCWA"])
best_idx = int(val_scwa.argmax())
print(
    f"Best SCWA achieved at epoch {best_idx + 1}: "
    f"SCWA={val_scwa[best_idx]:.4f}, "
    f"SWA={metrics['val_SWA'][best_idx]:.4f}, "
    f"CWA={metrics['val_CWA'][best_idx]:.4f}"
)
