import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---- organize stats ----
model_stats = {}
for model_name, stats in experiment_data.items():
    model_stats[model_name] = {
        "train_loss": np.array(stats["losses"]["train"]),
        "val_loss": np.array(stats["losses"]["val"]),
        "val_f1": np.array(stats["metrics"]["val"]),
        "epochs": np.arange(1, len(stats["losses"]["train"]) + 1),
    }

# identify dataset label for filenames
dataset_tag = "SPR_BENCH" if "SPR_BENCH" in working_dir else "synthetic_SPR"

# ---- 1: training loss curves ----
try:
    plt.figure()
    for m, d in model_stats.items():
        plt.plot(d["epochs"], d["train_loss"], label=f"{m}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_tag}: Training Loss vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_tag.lower()}_train_loss_models.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training-loss plot: {e}")
    plt.close()

# ---- 2: validation loss curves ----
try:
    plt.figure()
    for m, d in model_stats.items():
        plt.plot(d["epochs"], d["val_loss"], label=f"{m}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_tag}: Validation Loss vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_tag.lower()}_val_loss_models.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation-loss plot: {e}")
    plt.close()

# ---- 3: validation macro-F1 curves ----
try:
    plt.figure()
    for m, d in model_stats.items():
        plt.plot(d["epochs"], d["val_f1"], label=f"{m}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(f"{dataset_tag}: Validation Macro-F1 vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_tag.lower()}_val_f1_models.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation-F1 plot: {e}")
    plt.close()

# ---- 4: best macro-F1 bar plot ----
try:
    best_f1 = {m: d["val_f1"].max() for m, d in model_stats.items()}
    plt.figure()
    models, f1_vals = zip(*best_f1.items())
    plt.bar(range(len(models)), f1_vals, tick_label=list(models))
    plt.xlabel("Model")
    plt.ylabel("Best Val Macro-F1")
    plt.title(f"{dataset_tag}: Best Validation Macro-F1 by Model")
    fname = os.path.join(working_dir, f"{dataset_tag.lower()}_best_val_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best-F1 bar plot: {e}")
    plt.close()

# ---- numeric summary ----
for m, val in model_stats.items():
    print(f"{m}: best val Macro-F1 = {val['val_f1'].max():.4f}")
