import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("embedding_dim", {})
if not runs:
    print("No embedding_dim data found in experiment_data.npy")
    exit()

# collect final-epoch validation F1 for console print and bar plot
final_val_f1 = {k: v["metrics"]["val"][-1] for k, v in runs.items()}
print("Final validation F1 scores:")
for k, f1 in final_val_f1.items():
    print(f"  {k}: {f1:.4f}")

# --------------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    for k, v in runs.items():
        plt.plot(v["losses"]["train"], label=f"{k}_train")
        plt.plot(v["losses"]["val"], linestyle="--", label=f"{k}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Synthetic Dataset\nTraining vs Validation Loss (all embedding dims)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------------------------------------------------------------------
# 2) F1 curves
try:
    plt.figure()
    for k, v in runs.items():
        plt.plot(v["metrics"]["train"], label=f"{k}_train")
        plt.plot(v["metrics"]["val"], linestyle="--", label=f"{k}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(
        "SPR Synthetic Dataset\nTraining vs Validation Macro-F1 (all embedding dims)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# --------------------------------------------------------------------
# 3) Final F1 bar chart
try:
    plt.figure()
    keys = list(final_val_f1.keys())
    values = [final_val_f1[k] for k in keys]
    plt.bar(keys, values, color="skyblue")
    plt.ylabel("Validation Macro-F1 (final epoch)")
    plt.title("SPR Synthetic Dataset\nFinal Validation F1 by Embedding Dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_final_val_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()
