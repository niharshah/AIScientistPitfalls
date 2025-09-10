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

spr_logs = experiment_data.get("SPR_BENCH", {})
train_loss = spr_logs.get("losses", {}).get("train", [])
val_loss = spr_logs.get("losses", {}).get("val", [])
val_metrics = spr_logs.get("metrics", {}).get("val", [])

epochs = list(range(1, len(train_loss) + 1))
swa = [m["swa"] for m in val_metrics] if val_metrics else []
cwa = [m["cwa"] for m in val_metrics] if val_metrics else []
ccwa = [m["ccwa"] for m in val_metrics] if val_metrics else []

# -------- evaluation summary ---------
if ccwa:
    best_idx = int(np.argmax(ccwa))
    print(
        f"Best CCWA {ccwa[best_idx]:.4f} at epoch {best_idx+1} | "
        f"SWA={swa[best_idx]:.4f}, CWA={cwa[best_idx]:.4f}"
    )

# -------------- FIGURE 1: loss curves ---------------
try:
    if train_loss and val_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------- FIGURE 2: SWA -----------------------
try:
    if swa:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, swa, marker="o")
        plt.title("SPR_BENCH Validation SWA Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_SWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# -------------- FIGURE 3: CWA -----------------------
try:
    if cwa:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, cwa, marker="o", color="orange")
        plt.title("SPR_BENCH Validation CWA Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_CWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# -------------- FIGURE 4: CCWA ----------------------
try:
    if ccwa:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, ccwa, marker="o", color="green")
        plt.title("SPR_BENCH Validation CCWA Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_CCWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()

# -------------- FIGURE 5: best metrics bar ----------
try:
    if swa and cwa and ccwa:
        best_vals = [max(swa), max(cwa), max(ccwa)]
        labels = ["Best SWA", "Best CWA", "Best CCWA"]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, best_vals, color=["blue", "orange", "green"])
        plt.title("SPR_BENCH Best Validation Metrics")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_best_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating best metrics bar plot: {e}")
    plt.close()
