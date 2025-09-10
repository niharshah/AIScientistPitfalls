import matplotlib.pyplot as plt
import numpy as np
import os

# -------- paths and data loading --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
if ds_name not in experiment_data:
    print(f"{ds_name} not found in experiment_data, nothing to plot.")
    exit()

log = experiment_data[ds_name]
train_loss = [v for _, v in log["losses"]["train"]]
val_loss = [v for _, v in log["losses"]["val"]]
CWA = [m["CWA"] for _, m in log["metrics"]["val"]]
SWA = [m["SWA"] for _, m in log["metrics"]["val"]]
CpxWA = [m["CpxWA"] for _, m in log["metrics"]["val"]]
epochs = np.arange(1, len(train_loss) + 1)

# --------- 1) loss curves ------------
try:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, "o--", label="Train")
    plt.plot(epochs, val_loss, "s-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------- 2) metric curves ----------
try:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, CWA, "o-", label="CWA")
    plt.plot(epochs, SWA, "s-", label="SWA")
    plt.plot(epochs, CpxWA, "d-", label="CpxWA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Validation Accuracy Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_metric_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric curves plot: {e}")
    plt.close()

# --------- 3) final epoch bar chart ---
try:
    plt.figure(figsize=(6, 4))
    final_scores = [CWA[-1], SWA[-1], CpxWA[-1]]
    plt.bar(["CWA", "SWA", "CpxWA"], final_scores, color="skyblue")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Final Validation Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final metrics bar plot: {e}")
    plt.close()

# --------- print final metrics ----------
print(
    f"Final Validation Metrics -> CWA: {CWA[-1]:.3f}, SWA: {SWA[-1]:.3f}, CpxWA: {CpxWA[-1]:.3f}"
)
