import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = exp["frozen_embeddings"]["SPR_BENCH"]["num_epochs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}


# helper to get nice labels
def run_label(k):  # k like 'epochs_20'
    return k.replace("epochs_", "") + " epochs"


# 1) Validation loss curves
try:
    plt.figure()
    for k, v in runs.items():
        plt.plot(v["losses"]["val"], label=run_label(k))
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Frozen‐Embeddings GRU\nValidation Loss per Epoch (SPR_BENCH)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_frozen_emb_val_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val loss plot: {e}")
    plt.close()

# 2) Training loss curves
try:
    plt.figure()
    for k, v in runs.items():
        plt.plot(v["losses"]["train"], label=run_label(k))
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Frozen‐Embeddings GRU\nTraining Loss per Epoch (SPR_BENCH)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_frozen_emb_train_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating train loss plot: {e}")
    plt.close()

# 3) Validation HWA curves
try:
    plt.figure()
    for k, v in runs.items():
        hwa_vals = [m[2] for m in v["metrics"]["val"]]
        plt.plot(hwa_vals, label=run_label(k))
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("Frozen‐Embeddings GRU\nValidation HWA per Epoch (SPR_BENCH)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_frozen_emb_val_hwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val HWA plot: {e}")
    plt.close()

# 4) Test metric bar chart
try:
    labels = ["SWA", "CWA", "HWA"]
    x = np.arange(len(runs))  # bar groups
    width = 0.25
    plt.figure()
    for i, metric_idx in enumerate([0, 1, 2]):
        vals = [runs[k]["metrics"]["test"][metric_idx] for k in runs]
        plt.bar(x + i * width, vals, width, label=labels[i])
    plt.xticks(x + width, [run_label(k) for k in runs])
    plt.ylabel("Score")
    plt.title("Frozen‐Embeddings GRU\nTest Metrics by Training Epochs (SPR_BENCH)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_frozen_emb_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# 5) Train vs Val HWA for best overall run (highest final test HWA)
try:
    best_key = max(runs, key=lambda k: runs[k]["metrics"]["test"][2])
    best_run = runs[best_key]
    plt.figure()
    plt.plot([m[2] for m in best_run["metrics"]["train"]], label="Train HWA")
    plt.plot([m[2] for m in best_run["metrics"]["val"]], label="Val HWA")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title(f"Best Run ({run_label(best_key)})\nTrain vs Val HWA (SPR_BENCH)")
    plt.legend()
    fname = os.path.join(working_dir, f"spr_frozen_emb_best_{best_key}_hwa.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best run HWA plot: {e}")
    plt.close()
