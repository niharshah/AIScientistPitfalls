import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load every run listed in the prompt
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_d0670b7bc8ae43658fc047ce6dee1519_proc_3154574/experiment_data.npy",
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_e7ed40c2afe242c3a282c49c82b0ede2_proc_3154575/experiment_data.npy",
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_8a46dc6bee2a47149fb29438fae5e881_proc_3154576/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ------------------------------------------------------------------ #
# Aggregate per dataset
datasets = set()
for run_dict in all_experiment_data:
    datasets.update(run_dict.keys())

for dname in datasets:
    # --- collect runs that actually contain this dataset
    runs_for_ds = [run[dname] for run in all_experiment_data if dname in run]
    if not runs_for_ds:
        continue

    # ----- helper to stack metric arrays across runs -----
    def stack_metric(metric_path):
        """metric_path: tuple of nested keys, e.g. ('losses','train')"""
        collected = []
        for r in runs_for_ds:
            ref = r
            try:
                for k in metric_path:
                    ref = ref[k]
                collected.append(np.asarray(ref))
            except Exception:
                pass
        if not collected:
            return None
        # Trim to common length
        min_len = min([len(arr) for arr in collected])
        collected = np.stack([arr[:min_len] for arr in collected], axis=0)
        return collected

    # epoch axis
    epoch_arr = None
    try:
        epoch_candidates = [
            r.get("epochs", []) for r in runs_for_ds if r.get("epochs", [])
        ]
        if epoch_candidates:
            min_len = min(len(e) for e in epoch_candidates)
            epoch_arr = np.asarray(epoch_candidates[0][:min_len])
    except Exception:
        pass
    if epoch_arr is None:
        print(f"No epoch information for {dname}, skipping dataset")
        continue

    # -------------- Plot 1: Loss curves with SE -----------------
    try:
        train_loss = stack_metric(("losses", "train"))
        val_loss = stack_metric(("losses", "val"))
        if train_loss is not None and val_loss is not None:
            plt.figure()
            # train
            mean_tr = train_loss.mean(0)
            se_tr = train_loss.std(0, ddof=1) / np.sqrt(train_loss.shape[0])
            plt.plot(epoch_arr, mean_tr, label="Train Loss (mean)", color="tab:blue")
            plt.fill_between(
                epoch_arr,
                mean_tr - se_tr,
                mean_tr + se_tr,
                alpha=0.3,
                color="tab:blue",
                label="Train Loss (±SE)",
            )
            # val
            mean_val = val_loss.mean(0)
            se_val = val_loss.std(0, ddof=1) / np.sqrt(val_loss.shape[0])
            plt.plot(epoch_arr, mean_val, label="Val Loss (mean)", color="tab:orange")
            plt.fill_between(
                epoch_arr,
                mean_val - se_val,
                mean_val + se_val,
                alpha=0.3,
                color="tab:orange",
                label="Val Loss (±SE)",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname}: Train vs Validation Loss (Mean ± SE)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_agg_loss_curve.png")
            plt.savefig(fname)
            plt.close()
        else:
            print(f"{dname}: missing loss information, skipping loss plot")
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {dname}: {e}")
        plt.close()

    # -------------- Plot 2: Macro-F1 curves with SE -------------
    try:
        train_f1 = stack_metric(("metrics", "train_macro_f1"))
        val_f1 = stack_metric(("metrics", "val_macro_f1"))
        if train_f1 is not None and val_f1 is not None:
            plt.figure()
            # train
            mean_tr = train_f1.mean(0)
            se_tr = train_f1.std(0, ddof=1) / np.sqrt(train_f1.shape[0])
            plt.plot(
                epoch_arr, mean_tr, label="Train Macro-F1 (mean)", color="tab:green"
            )
            plt.fill_between(
                epoch_arr,
                mean_tr - se_tr,
                mean_tr + se_tr,
                alpha=0.3,
                color="tab:green",
                label="Train Macro-F1 (±SE)",
            )
            # val
            mean_val = val_f1.mean(0)
            se_val = val_f1.std(0, ddof=1) / np.sqrt(val_f1.shape[0])
            plt.plot(epoch_arr, mean_val, label="Val Macro-F1 (mean)", color="tab:red")
            plt.fill_between(
                epoch_arr,
                mean_val - se_val,
                mean_val + se_val,
                alpha=0.3,
                color="tab:red",
                label="Val Macro-F1 (±SE)",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dname}: Train vs Validation Macro-F1 (Mean ± SE)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_agg_macro_f1_curve.png")
            plt.savefig(fname)
            plt.close()

            # Print best mean val F1
            best_idx = int(np.argmax(mean_val))
            print(
                f"{dname} | Best mean Val Macro-F1: {mean_val[best_idx]:.4f} ± {se_val[best_idx]:.4f} at epoch {epoch_arr[best_idx]}"
            )
        else:
            print(f"{dname}: missing macro-F1 information, skipping F1 plot")
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated Macro-F1 curve for {dname}: {e}")
        plt.close()
