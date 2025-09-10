import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------#
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Retrieve SPR_BENCH results
spr_data = experiment_data.get("num_layers", {}).get("SPR_BENCH", {})
depth_keys = sorted(
    spr_data.keys(), key=lambda k: int(k.split("_")[1])
)  # e.g. 'layers_2' -> 2

# Aggregate metrics for printing
results_table = []

# ------------------------------------------------------------------#
# Plot 1: Loss curves
try:
    plt.figure(figsize=(6, 4))
    for dk in depth_keys:
        epochs = range(1, len(spr_data[dk]["losses"]["train"]) + 1)
        plt.plot(
            epochs, spr_data[dk]["losses"]["train"], label=f"{dk}-train", linestyle="-"
        )
        plt.plot(
            epochs, spr_data[dk]["losses"]["val"], label=f"{dk}-val", linestyle="--"
        )
    plt.title("SPR_BENCH: Training vs Validation Loss\n(Ablating Transformer Depth)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize="small", ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------#
# Plot 2: Accuracy curves
try:
    plt.figure(figsize=(6, 4))
    for dk in depth_keys:
        epochs = range(1, len(spr_data[dk]["metrics"]["train"]) + 1)
        plt.plot(
            epochs, spr_data[dk]["metrics"]["train"], label=f"{dk}-train", linestyle="-"
        )
        plt.plot(
            epochs, spr_data[dk]["metrics"]["val"], label=f"{dk}-val", linestyle="--"
        )
    plt.title(
        "SPR_BENCH: Training vs Validation Accuracy\n(Ablating Transformer Depth)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize="small", ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# ------------------------------------------------------------------#
# Prepare bar-plot data & compute test accuracy
val_final, test_final = [], []
for dk in depth_keys:
    val_acc = (
        spr_data[dk]["metrics"]["val"][-1] if spr_data[dk]["metrics"]["val"] else 0
    )
    preds = np.array(spr_data[dk]["predictions"])
    gts = np.array(spr_data[dk]["ground_truth"])
    test_acc = (preds == gts).mean() if len(gts) else 0
    val_final.append(val_acc)
    test_final.append(test_acc)
    results_table.append((dk, val_acc, test_acc))

# ------------------------------------------------------------------#
# Plot 3: Final validation accuracy bar chart
try:
    plt.figure(figsize=(6, 4))
    x = np.arange(len(depth_keys))
    plt.bar(x, val_final, color="steelblue")
    plt.xticks(x, [k.replace("layers_", "") for k in depth_keys])
    plt.title("SPR_BENCH: Final Validation Accuracy by Depth")
    plt.xlabel("Num Transformer Layers")
    plt.ylabel("Validation Accuracy")
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------#
# Print summary metrics
print("\nDepth | Final Val Acc | Test Acc")
for dk, v_acc, t_acc in results_table:
    n_layers = dk.replace("layers_", "")
    print(f"{n_layers:>5} | {v_acc:.4f}       | {t_acc:.4f}")
