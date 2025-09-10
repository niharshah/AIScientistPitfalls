import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper
def _safe(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# -------------- FIGURE 1: Pre-train loss -------------------
try:
    pre_losses = _safe(experiment_data, "pretrain", "loss")
    if pre_losses:
        epochs = [x["epoch"] for x in pre_losses]
        losses = [x["loss"] for x in pre_losses]
        plt.figure()
        plt.plot(epochs, losses, marker="o")
        plt.title("SPR_BENCH Pre-training Loss Curve\nLeft: Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        save_path = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating pretrain loss plot: {e}")
    plt.close()

# -------------- FIGURE 2: Fine-tune losses -----------------
try:
    tr_loss = _safe(experiment_data, "finetune", "losses", "train")
    val_loss = _safe(experiment_data, "finetune", "losses", "val")
    if tr_loss and val_loss:
        epochs = range(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title("SPR_BENCH Fine-tuning Loss\nLeft: Train vs. Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_finetune_losses.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating finetune loss plot: {e}")
    plt.close()

# -------------- FIGURE 3: Validation metrics ----------------
try:
    val_metrics = _safe(experiment_data, "finetune", "metrics", "val")
    if val_metrics:
        epochs = [m["epoch"] for m in val_metrics]
        swa = [m["swa"] for m in val_metrics]
        cwa = [m["cwa"] for m in val_metrics]
        ccwa = [m["ccwa"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, ccwa, label="CCWA")
        plt.title("SPR_BENCH Validation Metrics\nLeft: SWA, CWA, CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_validation_metrics.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# -------------- FIGURE 4: Correct vs Incorrect ----------------
try:
    preds = _safe(experiment_data, "finetune", "predictions")
    gts = _safe(experiment_data, "finetune", "ground_truth")
    if preds and gts and len(preds) == len(gts):
        correct = sum(p == t for p, t in zip(preds, gts))
        incorrect = len(preds) - correct
        plt.figure()
        plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["green", "red"])
        plt.title("SPR_BENCH Test Outcome\nLeft: Prediction Accuracy Snapshot")
        for i, v in enumerate([correct, incorrect]):
            plt.text(i, v + 0.5, str(v), ha="center")
        save_path = os.path.join(working_dir, "SPR_BENCH_correct_incorrect.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy bar plot: {e}")
    plt.close()

# -------------- PRINT BEST CCWA ----------------
best_ccwa = None
if _safe(experiment_data, "finetune", "metrics", "val"):
    best_ccwa = max(m["ccwa"] for m in experiment_data["finetune"]["metrics"]["val"])
print(f"Best CCWA achieved: {best_ccwa}")
