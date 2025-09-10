import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ paths & data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp, depths = None, []
else:
    exp = exp["n_layers"]["SPR_BENCH"]
    depths = exp["depths"]

# ------------------------------------------------------------------ quick summary
if depths:
    train_f1_final = exp["metrics"]["train_f1"]
    val_f1_final = exp["metrics"]["val_f1"]
    best_idx = int(np.argmax(val_f1_final))
    print("Depths :", depths)
    print("Final train F1 :", [f"{v:.4f}" for v in train_f1_final])
    print("Final  val  F1 :", [f"{v:.4f}" for v in val_f1_final])
    print(
        f"Best depth={depths[best_idx]} with Dev Macro F1={val_f1_final[best_idx]:.4f}"
    )

# ------------------------------------------------------------------ per-depth curves (≤5 figs)
for depth in depths:
    try:
        curve = exp["epoch_curves"][depth]
        epochs = range(1, len(curve["train_loss"]) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # Left subplot: loss
        ax[0].plot(epochs, curve["train_loss"], label="train_loss")
        ax[0].plot(epochs, curve["val_loss"], label="val_loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Left: Loss")
        ax[0].legend()
        # Right subplot: F1
        ax[1].plot(epochs, curve["train_f1"], label="train_F1")
        ax[1].plot(epochs, curve["val_f1"], label="val_F1")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Macro-F1")
        ax[1].set_title("Right: Macro-F1")
        ax[1].legend()
        fig.suptitle(f"SPR_BENCH TinyTransformer n_layers={depth} Training Curves")
        fname = f"SPR_BENCH_curves_depth_{depth}.png"
        fig.savefig(os.path.join(working_dir, fname))
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting curves for depth {depth}: {e}")
        plt.close()

# ------------------------------------------------------------------ aggregated bar plot (1 fig)
try:
    if depths:
        fig = plt.figure()
        x_pos = np.arange(len(depths))
        plt.bar(x_pos, val_f1_final, alpha=0.7, color="tab:blue")
        plt.xticks(x_pos, depths)
        plt.xlabel("Number of Transformer Layers")
        plt.ylabel("Final Validation Macro-F1")
        plt.title("SPR_BENCH Validation Macro-F1 vs Depth")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_valF1_vs_depth.png"))
        plt.close(fig)
except Exception as e:
    print(f"Error creating aggregated F1 plot: {e}")
    plt.close()
