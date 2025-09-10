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

model_key = "NoProjectionHeadContrastive"
data_key = "SPR_BENCH"
data_dict = experiment_data.get(model_key, {}).get(data_key, {})

train_loss = data_dict.get("losses", {}).get("train", [])
val_loss = data_dict.get("losses", {}).get("val", [])
val_metric = data_dict.get("metrics", {}).get("val", [])

best_val_loss = min(val_loss) if val_loss else None
best_val_metric = max(val_metric) if val_metric else None

# ----------------- plot loss curves -----------------
try:
    plt.figure()
    epochs = range(1, len(val_loss) + 1)
    if train_loss:
        plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ----------------- plot CompWA curve -----------------
try:
    if val_metric:
        plt.figure()
        plt.plot(range(1, len(val_metric) + 1), val_metric, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Comp-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation CompWA Curve")
        fname = os.path.join(working_dir, "SPR_BENCH_compwa_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CompWA plot: {e}")
    plt.close()

print(f"Best Validation Loss   : {best_val_loss}")
print(f"Best Validation CompWA : {best_val_metric}")
