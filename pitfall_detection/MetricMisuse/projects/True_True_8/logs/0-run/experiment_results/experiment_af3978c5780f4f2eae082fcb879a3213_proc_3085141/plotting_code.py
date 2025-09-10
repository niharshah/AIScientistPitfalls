import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- preparation ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

run_name = "contrastive+finetune"
run_store = experiment_data.get(run_name, {})
dataset_name = "synthetic_SPR"


def unpack(store, path):
    """returns epochs, values arrays for a given path tuple"""
    for p in path:
        store = store[p]
    if not store:
        return np.array([]), np.array([])
    ep, val = zip(*store)
    return np.array(ep), np.array(val)


plot_idx = 0
max_plots = 5

# ----------------------- 1) loss curve -----------------------
if plot_idx < max_plots:
    try:
        ep_tr, tr = unpack(run_store, ("losses", "train"))
        ep_val, val = unpack(run_store, ("losses", "val"))
        plt.figure()
        plt.plot(ep_tr, tr, label="Train")
        plt.plot(ep_val, val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Loss Curves ({dataset_name})\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()
    plot_idx += 1

# ----------------------- 2) SWA curve ------------------------
if plot_idx < max_plots:
    try:
        ep_swa, swa_vals = unpack(run_store, ("metrics", "SWA"))
        plt.figure()
        plt.plot(ep_swa, swa_vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.title(
            f"SWA over Epochs ({dataset_name})\nLeft: Ground Truth, Right: Predictions"
        )
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_SWA_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
        plt.close()
    plot_idx += 1

# ----------------------- 3) CWA curve ------------------------
if plot_idx < max_plots:
    try:
        ep_cwa, cwa_vals = unpack(run_store, ("metrics", "CWA"))
        plt.figure()
        plt.plot(ep_cwa, cwa_vals, marker="s", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.title(
            f"CWA over Epochs ({dataset_name})\nLeft: Ground Truth, Right: Predictions"
        )
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_CWA_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA curve: {e}")
        plt.close()
    plot_idx += 1

# ----------------------- 4) CompWA curve ---------------------
if plot_idx < max_plots:
    try:
        ep_comp, comp_vals = unpack(run_store, ("metrics", "CompWA"))
        plt.figure()
        plt.plot(ep_comp, comp_vals, marker="^", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title(f"CompWA over Epochs ({dataset_name})")
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_CompWA_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA curve: {e}")
        plt.close()
    plot_idx += 1

# ----------------------- 5) final metrics bar ----------------
if plot_idx < max_plots:
    try:
        final_vals = [
            swa_vals[-1] if len(swa_vals) else 0,
            cwa_vals[-1] if len(cwa_vals) else 0,
            comp_vals[-1] if len(comp_vals) else 0,
        ]
        labels = ["SWA", "CWA", "CompWA"]
        x = np.arange(len(labels))
        plt.figure()
        plt.bar(x, final_vals, color=["steelblue", "orange", "green"])
        plt.xticks(x, labels)
        plt.ylabel("Score")
        plt.title(f"Final Metric Values ({dataset_name})")
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_final_metrics_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()
