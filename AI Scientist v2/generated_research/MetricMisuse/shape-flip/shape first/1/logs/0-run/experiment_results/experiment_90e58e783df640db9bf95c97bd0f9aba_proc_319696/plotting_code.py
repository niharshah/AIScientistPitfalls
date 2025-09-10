import matplotlib.pyplot as plt
import numpy as np
import os

# working directory ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
# load experiment data -------------------------------------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ----------------------------------------------------------------------
# helper to fetch metrics ----------------------------------------------
def get_hidden_runs(exp_dict):
    runs = exp_dict.get("frozen_embedding_ablation", {}).get("SPR_BENCH", {})
    return {k: v for k, v in runs.items()}


runs = get_hidden_runs(experiment_data)

summary = {}  # store metrics to print later

# ----------------------------------------------------------------------
# per-run accuracy / loss curves ---------------------------------------
for run_name, mdata in runs.items():
    epochs = list(range(1, len(mdata["metrics"]["train_acc"]) + 1))
    # -------- accuracy plot ----------
    try:
        plt.figure()
        plt.plot(epochs, mdata["metrics"]["train_acc"], label="Train")
        plt.plot(epochs, mdata["metrics"]["val_acc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"SPR_BENCH Accuracy Curves ({run_name})\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{run_name}_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {run_name}: {e}")
        plt.close()
    # -------- loss plot ----------
    try:
        plt.figure()
        plt.plot(epochs, mdata["losses"]["train"], label="Train")
        plt.plot(epochs, mdata["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH Loss Curves ({run_name})\nTrain vs Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{run_name}_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {run_name}: {e}")
        plt.close()
    # gather summary numbers
    last_val_acc = (
        mdata["metrics"]["val_acc"][-1] if mdata["metrics"]["val_acc"] else np.nan
    )
    zsrta = mdata["metrics"]["ZSRTA"][0] if mdata["metrics"]["ZSRTA"] else np.nan
    summary[run_name] = (last_val_acc, zsrta)

# ----------------------------------------------------------------------
# summary ZSRTA bar plot -----------------------------------------------
try:
    plt.figure()
    names = list(summary.keys())
    zs_vals = [summary[n][1] for n in names]
    plt.bar(range(len(names)), zs_vals, tick_label=names)
    plt.ylabel("ZSRTA")
    plt.title("SPR_BENCH Zero-Shot Rule Transfer Accuracy\nAcross Hidden Dimensions")
    fname = os.path.join(working_dir, f"SPR_BENCH_ZSRTA_comparison.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating ZSRTA bar plot: {e}")
    plt.close()

# ----------------------------------------------------------------------
# print summary --------------------------------------------------------
print("\n=== Validation Accuracy & ZSRTA Summary ===")
for run_name, (val_acc, zsrta) in summary.items():
    print(f"{run_name:15s} | ValAcc: {val_acc:.4f} | ZSRTA: {zsrta:.4f}")
