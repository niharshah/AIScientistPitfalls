import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load all experiment files -----------------------
experiment_data_path_list = [
    "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_8b8fb0f31ed84a998728738ff28b407f_proc_1723260/experiment_data.npy",
    "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_26ef9177e5ca4b56a0f24fe36acd1e60_proc_1723258/experiment_data.npy",
    "experiments/2025-08-31_14-12-13_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_4dcc46c59d734728a7be20e348b98260_proc_1723259/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", os.getcwd())
    for p in experiment_data_path_list:
        fp = os.path.join(root, p)
        all_experiment_data.append(np.load(fp, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ---------------- aggregate across runs ------------------------------
aggregated = {}
for exp in all_experiment_data:
    for ds_name, ds_data in exp.items():
        ds_agg = aggregated.setdefault(
            ds_name, {"losses": {"train": [], "val": []}, "CompWA": []}
        )
        # losses
        ds_agg["losses"]["train"].append(
            np.array(ds_data.get("losses", {}).get("train", []))
        )
        ds_agg["losses"]["val"].append(
            np.array(ds_data.get("losses", {}).get("val", []))
        )
        # CompWA
        if "metrics" in ds_data and "val_CompWA" in ds_data["metrics"]:
            ds_agg["CompWA"].append(np.array(ds_data["metrics"]["val_CompWA"]))


# ---------------- helper ------------------------------------------------
def close_fig():
    if plt.get_fignums():
        plt.close()


# ---------------- create plots -----------------------------------------
for ds_name, ds_data in aggregated.items():
    # ---------- aggregated loss curve -----------------------------------
    try:
        train_runs = [arr for arr in ds_data["losses"]["train"] if len(arr)]
        val_runs = [arr for arr in ds_data["losses"]["val"] if len(arr)]
        if train_runs and val_runs:
            min_len = min(min(map(len, train_runs)), min(map(len, val_runs)))
            train_stack = np.stack([a[:min_len] for a in train_runs], axis=0)
            val_stack = np.stack([a[:min_len] for a in val_runs], axis=0)
            epochs = np.arange(1, min_len + 1)

            train_mean, train_se = train_stack.mean(0), train_stack.std(
                0, ddof=1
            ) / np.sqrt(train_stack.shape[0])
            val_mean, val_se = val_stack.mean(0), val_stack.std(0, ddof=1) / np.sqrt(
                val_stack.shape[0]
            )

            plt.figure()
            plt.plot(epochs, train_mean, label="Train Mean", color="tab:blue")
            plt.fill_between(
                epochs,
                train_mean - train_se,
                train_mean + train_se,
                alpha=0.3,
                color="tab:blue",
                label="Train ±1SE",
            )
            plt.plot(epochs, val_mean, label="Val Mean", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_se,
                val_mean + val_se,
                alpha=0.3,
                color="tab:orange",
                label="Val ±1SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.title(
                f"{ds_name} Aggregated Loss Curve\nMean ±1SE over {train_stack.shape[0]} runs"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_aggregated_loss_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        else:
            print(f"No loss data found for {ds_name}")
    except Exception as e:
        print(f"Error creating aggregated loss curve for {ds_name}: {e}")
    finally:
        close_fig()

    # ---------- aggregated CompWA ---------------------------------------
    try:
        compwa_runs = [arr for arr in ds_data["CompWA"] if len(arr)]
        if compwa_runs:
            min_len = min(map(len, compwa_runs))
            compwa_stack = np.stack([a[:min_len] for a in compwa_runs], axis=0)
            epochs = np.arange(1, min_len + 1)

            compwa_mean = compwa_stack.mean(0)
            compwa_se = compwa_stack.std(0, ddof=1) / np.sqrt(compwa_stack.shape[0])

            plt.figure()
            plt.plot(epochs, compwa_mean, label="Val CompWA Mean", color="green")
            plt.fill_between(
                epochs,
                compwa_mean - compwa_se,
                compwa_mean + compwa_se,
                alpha=0.3,
                color="green",
                label="±1SE",
            )
            plt.ylim(0, 1)
            plt.xlabel("Epoch")
            plt.ylabel("CompWA")
            plt.title(
                f"{ds_name} Aggregated Validation CompWA\nMean ±1SE over {compwa_stack.shape[0]} runs"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_aggregated_val_CompWA.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            print(
                f"{ds_name} final epoch CompWA mean: {compwa_mean[-1]:.4f} ± {compwa_se[-1]:.4f}"
            )
        else:
            print(f"No CompWA data found for {ds_name}")
    except Exception as e:
        print(f"Error creating aggregated CompWA plot for {ds_name}: {e}")
    finally:
        close_fig()
