import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# shortcut handles
bench_key = "nlayers_tuning"
dataset_key = "SPR_BENCH"
runs = experiment_data.get(bench_key, {}).get(dataset_key, {})

# aggregate per-run metrics
layer_ids, train_losses, val_losses, train_f1s, val_f1s, final_val_f1 = (
    [],
    [],
    [],
    [],
    [],
    [],
)
for cfg_name, run_dict in runs.items():  # e.g. cfg_name = 'nlayers_3'
    nlayers = int(cfg_name.split("_")[-1])
    layer_ids.append(nlayers)
    tr_loss = [e["loss"] for e in run_dict["losses"]["train"]]
    vl_loss = [e["loss"] for e in run_dict["losses"]["val"]]
    tr_f1 = [e["macro_f1"] for e in run_dict["metrics"]["train"]]
    vl_f1 = [e["macro_f1"] for e in run_dict["metrics"]["val"]]
    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    train_f1s.append(tr_f1)
    val_f1s.append(vl_f1)
    final_val_f1.append(vl_f1[-1] if vl_f1 else np.nan)

# sort by nlayers so plots look orderly
order = np.argsort(layer_ids)
layer_ids = np.array(layer_ids)[order]
train_losses = np.array(train_losses, dtype=object)[order]
val_losses = np.array(val_losses, dtype=object)[order]
train_f1s = np.array(train_f1s, dtype=object)[order]
val_f1s = np.array(val_f1s, dtype=object)[order]
final_val_f1 = np.array(final_val_f1)[order]

# ----------------- plotting helpers -----------------
epochs = np.arange(1, len(train_losses[0]) + 1) if len(train_losses) else []

# 1) Loss curves
try:
    plt.figure(figsize=(7, 4))
    for nl, tr, vl in zip(layer_ids, train_losses, val_losses):
        plt.plot(epochs, tr, "--", label=f"Train nl={nl}")
        plt.plot(epochs, vl, "-", label=f"Val nl={nl}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss across Transformer Depths")
    plt.legend(fontsize=7, ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_nlayers.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Macro-F1 curves
try:
    plt.figure(figsize=(7, 4))
    for nl, tr, vl in zip(layer_ids, train_f1s, val_f1s):
        plt.plot(epochs, tr, "--", label=f"Train nl={nl}")
        plt.plot(epochs, vl, "-", label=f"Val nl={nl}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation Macro-F1 across Transformer Depths")
    plt.legend(fontsize=7, ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves_nlayers.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# 3) Final validation F1 bar chart
try:
    plt.figure(figsize=(6, 4))
    plt.bar(layer_ids, final_val_f1, color="skyblue")
    plt.xlabel("Number of Transformer Layers")
    plt.ylabel("Final Epoch Validation Macro-F1")
    plt.title("SPR_BENCH: Final Val Macro-F1 vs Transformer Depth")
    for x, y in zip(layer_ids, final_val_f1):
        plt.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_final_val_f1_bar.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

print("Plots saved to:", working_dir)
