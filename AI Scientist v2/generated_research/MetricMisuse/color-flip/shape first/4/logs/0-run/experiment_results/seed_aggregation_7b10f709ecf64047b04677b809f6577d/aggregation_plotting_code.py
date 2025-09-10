import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load every experiment_data.npy -------------
experiment_data_path_list = [
    "experiments/2025-08-15_23-37-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_71f03f1e1c414307878ea9311c9d304c_proc_3032290/experiment_data.npy",
    "experiments/2025-08-15_23-37-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_08019eacd8d1436e9c0b5a160f34bc67_proc_3032292/experiment_data.npy",
    "experiments/2025-08-15_23-37-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_1875cac921634a5a90d9441d3231c964_proc_3032291/experiment_data.npy",
]
all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for p in experiment_data_path_list:
        full_path = os.path.join(root, p)
        if os.path.exists(full_path):
            d = np.load(full_path, allow_pickle=True).item()
            all_experiment_data.append(d)
        else:
            print(f"Path not found, skipping: {full_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")


# -------- helper: gather per-dataset curves ----------
def aggregate_curves(dataset_key, key_chain):
    """
    key_chain: list like ['losses','train']  OR  ['metrics','val']
    returns stacked_array (runs x epochs) truncated to min len
    """
    curves = []
    for exp in all_experiment_data:
        try:
            runs = exp["epochs"][dataset_key]
            for _, rec in runs.items():
                cur = rec
                for k in key_chain:
                    cur = cur[k]
                curves.append(np.asarray(cur, dtype=float))
        except Exception:
            continue
    if not curves:
        return None
    min_len = min(len(c) for c in curves)
    trimmed = np.stack([c[:min_len] for c in curves], axis=0)
    return trimmed  # shape (n_runs, min_len)


# --------- list all datasets present ----------
datasets = set()
for exp in all_experiment_data:
    try:
        datasets.update(exp["epochs"].keys())
    except Exception:
        continue

# ------------- plotting per dataset ---------------
for ds in datasets:
    # ---- Plot 1: aggregated loss curves -----------
    try:
        train_mat = aggregate_curves(ds, ["losses", "train"])
        val_mat = aggregate_curves(ds, ["losses", "val"])
        if train_mat is not None and val_mat is not None:
            epochs = np.arange(1, train_mat.shape[1] + 1)
            train_mean = train_mat.mean(axis=0)
            train_se = train_mat.std(axis=0, ddof=1) / np.sqrt(train_mat.shape[0])
            val_mean = val_mat.mean(axis=0)
            val_se = val_mat.std(axis=0, ddof=1) / np.sqrt(val_mat.shape[0])

            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs, train_mean, label="Train mean", linestyle="--", color="tab:blue"
            )
            plt.fill_between(
                epochs,
                train_mean - train_se,
                train_mean + train_se,
                alpha=0.3,
                color="tab:blue",
                label="Train ± SE",
            )
            plt.plot(epochs, val_mean, label="Val mean", color="tab:orange")
            plt.fill_between(
                epochs,
                val_mean - val_se,
                val_mean + val_se,
                alpha=0.3,
                color="tab:orange",
                label="Val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds}: Training vs Validation Loss Curves (mean ± SE)")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_loss_curves_agg.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        else:
            print(f"No data to plot aggregated loss for {ds}")
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds}: {e}")
    finally:
        plt.close()

    # ---- Plot 2: aggregated validation HWA --------
    try:
        hwa_mat = aggregate_curves(ds, ["metrics", "val"])
        if hwa_mat is not None:
            epochs = np.arange(1, hwa_mat.shape[1] + 1)
            hwa_mean = hwa_mat.mean(axis=0)
            hwa_se = hwa_mat.std(axis=0, ddof=1) / np.sqrt(hwa_mat.shape[0])

            plt.figure(figsize=(7, 5))
            plt.plot(epochs, hwa_mean, label="Val HWA mean", color="tab:green")
            plt.fill_between(
                epochs,
                hwa_mean - hwa_se,
                hwa_mean + hwa_se,
                alpha=0.3,
                color="tab:green",
                label="Val HWA ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Harmonic Weighted Accuracy")
            plt.title(f"{ds}: Validation HWA Across Epochs (mean ± SE)")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_HWA_curves_agg.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        else:
            print(f"No data to plot aggregated HWA for {ds}")
    except Exception as e:
        print(f"Error creating aggregated HWA plot for {ds}: {e}")
    finally:
        plt.close()
