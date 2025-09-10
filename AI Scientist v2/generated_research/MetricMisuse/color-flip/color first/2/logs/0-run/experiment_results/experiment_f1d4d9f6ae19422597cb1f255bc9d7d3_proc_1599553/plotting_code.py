import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("SPR_BENCH", {})

epochs = spr_data.get("epochs", [])
train_losses = spr_data.get("losses", {}).get("train", [])
val_losses = spr_data.get("losses", {}).get("val", [])
train_metrics = spr_data.get("metrics", {}).get("train", [])
val_metrics = spr_data.get("metrics", {}).get("val", [])


# Unpack each metric if available
def extract_metric(idx, split_metrics):
    return [m[idx] for m in split_metrics] if split_metrics else []


cwa_tr, swa_tr, gcwa_tr = (extract_metric(i, train_metrics) for i in range(3))
cwa_val, swa_val, gcwa_val = (extract_metric(i, val_metrics) for i in range(3))


# ------------------------------------------------------------------ #
# Helper for plotting
def safe_curve_plot(x, y1, y2, ylabel, title, fname):
    try:
        if not x or not y1 or not y2:
            raise ValueError("Missing data to plot.")
        plt.figure(figsize=(6, 4))
        plt.plot(x, y1, label="train")
        plt.plot(x, y2, label="val")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating {fname}: {e}")
    finally:
        plt.close()


# ------------------------------------------------------------------ #
# 1) Loss curve
safe_curve_plot(
    epochs,
    train_losses,
    val_losses,
    "Loss",
    "SPR_BENCH: Training vs Validation Loss",
    "spr_bench_loss_curve.png",
)

# 2) Color-Weighted Accuracy curve
safe_curve_plot(
    epochs,
    cwa_tr,
    cwa_val,
    "CWA",
    "SPR_BENCH: Color-Weighted Accuracy",
    "spr_bench_cwa_curve.png",
)

# 3) Shape-Weighted Accuracy curve
safe_curve_plot(
    epochs,
    swa_tr,
    swa_val,
    "SWA",
    "SPR_BENCH: Shape-Weighted Accuracy",
    "spr_bench_swa_curve.png",
)

# 4) Glyph Complexity-Weighted Accuracy curve
safe_curve_plot(
    epochs,
    gcwa_tr,
    gcwa_val,
    "GCWA",
    "SPR_BENCH: Glyph Complexity-Weighted Accuracy",
    "spr_bench_gcwa_curve.png",
)

# ------------------------------------------------------------------ #
# Compute simple test accuracy if predictions & ground_truth exist
preds = np.array(spr_data.get("predictions", []))
truth = np.array(spr_data.get("ground_truth", []))
test_acc = (preds == truth).mean() if preds.size and truth.size else None
print(
    f"Test Accuracy: {test_acc:.4f}" if test_acc is not None else "Test Accuracy: N/A"
)
