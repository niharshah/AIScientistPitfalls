import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- LOAD DATA --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Utility: extract metrics and also keep best val F1
best_f1_table = {}  # {(model, bs): f1}

# -------------------- PER-MODEL FIGURES --------------------
for abl_name, abl_dict in experiment_data.items():
    try:
        plt.figure(figsize=(10, 4))
        # left subplot: losses
        plt.subplot(1, 2, 1)
        for bs, run in abl_dict["spr_bench"]["batch_size"].items():
            epochs = run["epochs"]
            plt.plot(
                epochs, run["losses"]["train"], label=f"train bs={bs}", linestyle="--"
            )
            plt.plot(epochs, run["losses"]["val"], label=f"val bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Left: Train vs Val Loss")
        plt.legend(fontsize=6)

        # right subplot: val F1
        plt.subplot(1, 2, 2)
        for bs, run in abl_dict["spr_bench"]["batch_size"].items():
            epochs = run["epochs"]
            val_f1 = run["metrics"]["val_f1"]
            plt.plot(epochs, val_f1, label=f"bs={bs}")
            best_f1_table[(abl_name, bs)] = max(val_f1)
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Right: Validation Macro-F1")
        plt.legend(fontsize=6)

        plt.suptitle(f"{abl_name} on spr_bench")
        fname = os.path.join(working_dir, f"spr_bench_{abl_name}_loss_f1_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating {abl_name} figure: {e}")
        plt.close()

# -------------------- BAR CHART OF BEST F1 --------------------
try:
    plt.figure(figsize=(6, 4))
    models = sorted({k[0] for k in best_f1_table})
    bss = sorted({k[1] for k in best_f1_table})
    width = 0.35
    x = np.arange(len(bss))
    for i, model in enumerate(models):
        vals = [best_f1_table.get((model, bs), 0) for bs in bss]
        plt.bar(x + i * width, vals, width=width, label=model)
    plt.xticks(x + width / 2, [str(bs) for bs in bss])
    plt.xlabel("Batch Size")
    plt.ylabel("Best Validation Macro-F1")
    plt.title("spr_bench: Best Val F1 per Batch Size")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_best_val_f1_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating best-F1 bar chart: {e}")
    plt.close()

# -------------------- PRINT SUMMARY --------------------
print("\nBest Val Macro-F1 Scores")
for (model, bs), score in sorted(best_f1_table.items()):
    print(f"{model:12s} | batch_size={bs:3d} | best_val_f1={score:.4f}")
