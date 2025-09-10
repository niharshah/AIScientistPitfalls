import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- set up -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load all experiment_data.npy --------
experiment_data_path_list = [
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_f3a89b2e29f9428dbe2bd53139c15b3d_proc_3071489/experiment_data.npy",
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_1015f05b31df4338b08e899490c83d7c_proc_3071487/experiment_data.npy",
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_8df468bebbf549a79b2d3fd4f5dd4c29_proc_3071488/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if len(all_experiment_data) == 0:
    print("No experiment data could be loaded; aborting plots.")
    exit()

# assume same dataset name across runs
dataset_name = list(all_experiment_data[0].get("embed_dim_tuning", {}).values())[0].get(
    "dataset_name", "dataset"
)


# ---------------- helper ----------------------
def unpack(run_store, path):
    """path like ('losses','train') returns epoch list, value list"""
    items = run_store
    for p in path:
        items = items[p]
    epochs, vals = zip(*items)
    return np.array(epochs), np.array(vals)


# Collect union of all embedding keys
embed_keys = sorted(
    {k for exp in all_experiment_data for k in exp.get("embed_dim_tuning", {})}
)

# -------------------------------------------------------
# 1) Aggregated train / val loss with stderr
# -------------------------------------------------------
try:
    for loss_split, color in [("train", "tab:blue"), ("val", "tab:orange")]:
        plt.figure()
        for k in embed_keys:
            stacks = []
            for exp in all_experiment_data:
                tuning = exp.get("embed_dim_tuning", {})
                if k not in tuning:  # key missing in this run
                    break
            else:
                # every run has this key
                epochs = None
                for exp in all_experiment_data:
                    ep, vals = unpack(
                        exp["embed_dim_tuning"][k], ("losses", loss_split)
                    )
                    if epochs is None:
                        epochs = ep
                    stacks.append(vals)
                mat = np.vstack(stacks)
                mean = mat.mean(axis=0)
                stderr = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
                plt.plot(epochs, mean, label=f"{k.split('_')[1]}d mean")
                plt.fill_between(epochs, mean - stderr, mean + stderr, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{loss_split.capitalize()} Loss (Aggregated) ({dataset_name})")
        plt.legend(title="Embedding dim")
        fname = f"{dataset_name}_{loss_split}_loss_aggregated.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plots: {e}")
    plt.close()

# -------------------------------------------------------
# 2) Aggregated CoWA vs. epoch
# -------------------------------------------------------
try:
    plt.figure()
    for k in embed_keys:
        stacks = []
        for exp in all_experiment_data:
            tuning = exp.get("embed_dim_tuning", {})
            if k not in tuning:
                break
        else:
            epochs = None
            for exp in all_experiment_data:
                ep, vals = unpack(exp["embed_dim_tuning"][k], ("metrics", "CoWA"))
                if epochs is None:
                    epochs = ep
                stacks.append(vals)
            mat = np.vstack(stacks)
            mean = mat.mean(0)
            stderr = mat.std(0, ddof=1) / np.sqrt(mat.shape[0])
            plt.plot(epochs, mean, label=f"{k.split('_')[1]}d mean")
            plt.fill_between(epochs, mean - stderr, mean + stderr, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("CoWA")
    plt.title(
        f"CoWA over Epochs (Aggregated) ({dataset_name})\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.legend(title="Embed dim")
    fname = f"{dataset_name}_CoWA_epochs_aggregated.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CoWA plot: {e}")
    plt.close()

# -------------------------------------------------------
# 3) Final-epoch CoWA bar chart with error bars
# -------------------------------------------------------
final_results = {}
try:
    dims, means, errs = [], [], []
    for k in embed_keys:
        finals = []
        for exp in all_experiment_data:
            tuning = exp.get("embed_dim_tuning", {})
            if k not in tuning:
                break
        else:
            for exp in all_experiment_data:
                finals.append(
                    unpack(exp["embed_dim_tuning"][k], ("metrics", "CoWA"))[1][-1]
                )
            finals = np.array(finals)
            dim = k.split("_")[1]
            dims.append(dim)
            means.append(finals.mean())
            errs.append(finals.std(ddof=1) / np.sqrt(len(finals)))
            final_results[dim] = (means[-1], errs[-1])
    x = np.arange(len(dims))
    plt.figure()
    plt.bar(x, means, yerr=errs, capsize=5, color="skyblue", alpha=0.8)
    plt.xticks(x, dims)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Final CoWA")
    plt.title(f"Final CoWA by Embedding Size (Aggregated) ({dataset_name})")
    fname = f"{dataset_name}_final_CoWA_bar_aggregated.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final CoWA bar chart: {e}")
    plt.close()

# ------------------- print summary --------------------
print("Final CoWA (mean ± stderr):")
for dim, (m, s) in final_results.items():
    print(f"  {dim}d : {m:.4f} ± {s:.4f}")
