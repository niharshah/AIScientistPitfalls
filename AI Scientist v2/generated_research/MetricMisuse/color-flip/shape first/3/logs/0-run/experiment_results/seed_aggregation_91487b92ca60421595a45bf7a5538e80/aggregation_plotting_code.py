import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------- basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------- load all runs
experiment_data_path_list = [
    "None/experiment_data.npy",
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_931924c2e32446af960de0b63e912856_proc_3017223/experiment_data.npy",
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_532bd0b4a1534b54800930dece9c53c5_proc_3017225/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_d = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(exp_d)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

if not all_experiment_data:
    print("No experiment data loaded; aborting plotting.")
    exit()


# -------------------------------------------------------- helper utilities
def aligned_stack(list_of_arrays):
    """Stack 1-D arrays after truncating them to the minimum length."""
    if not list_of_arrays:
        return None
    min_len = min(len(a) for a in list_of_arrays)
    trimmed = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    return trimmed  # shape (n_runs, min_len)


def mean_and_sem(stacked_arr):
    """Return mean and standard error along axis 0."""
    if stacked_arr is None:
        return None, None
    mean = stacked_arr.mean(axis=0)
    sem = stacked_arr.std(axis=0, ddof=1) / np.sqrt(stacked_arr.shape[0])
    return mean, sem


# -------------------------------------------------------- aggregate per-dataset
datasets = set()
for run in all_experiment_data:
    datasets.update(run.get("num_layers", {}).keys())

color_cycle = ["r", "g", "b", "m", "c", "y", "k"]

for ds_name in datasets:
    # collect per-layer information across runs
    per_layer_runs = {}
    for run in all_experiment_data:
        nl_dict = run.get("num_layers", {}).get(ds_name, {}).get("per_layer", {})
        for layer, rec in nl_dict.items():
            per_layer_runs.setdefault(
                layer,
                {
                    "train_loss": [],
                    "val_loss": [],
                    "val_scwa": [],
                    "best_val": [],
                    "test": [],
                },
            )
            per_layer_runs[layer]["train_loss"].append(np.array(rec["losses"]["train"]))
            per_layer_runs[layer]["val_loss"].append(np.array(rec["losses"]["val"]))
            per_layer_runs[layer]["val_scwa"].append(np.array(rec["metrics"]["val"]))
            per_layer_runs[layer]["best_val"].append(rec["best_val_scwa"])
            per_layer_runs[layer]["test"].append(
                run["num_layers"][ds_name].get("test_scwa")
                if run["num_layers"][ds_name].get("best_layer") == layer
                else np.nan
            )

    sorted_layers = sorted(per_layer_runs.keys())
    colors = {l: color_cycle[i % len(color_cycle)] for i, l in enumerate(sorted_layers)}

    # ---------------------------------------------------- Plot 1: Loss curves with error bands
    try:
        plt.figure()
        for l in sorted_layers:
            train_stack = aligned_stack(per_layer_runs[l]["train_loss"])
            val_stack = aligned_stack(per_layer_runs[l]["val_loss"])
            if train_stack is None or val_stack is None:
                continue
            train_mean, train_sem = mean_and_sem(train_stack)
            val_mean, val_sem = mean_and_sem(val_stack)
            epochs = np.arange(len(train_mean))
            plt.plot(
                epochs, train_mean, linestyle="--", color=colors[l], label=f"Train L{l}"
            )
            plt.fill_between(
                epochs,
                train_mean - train_sem,
                train_mean + train_sem,
                color=colors[l],
                alpha=0.15,
            )
            plt.plot(
                epochs, val_mean, linestyle="-", color=colors[l], label=f"Val L{l}"
            )
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                color=colors[l],
                alpha=0.15,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"{ds_name}: Mean±SEM Training & Validation Loss")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_loss_mean_sem.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ---------------------------------------------------- Plot 2: Validation SCWA with error bands
    try:
        plt.figure()
        for l in sorted_layers:
            scwa_stack = aligned_stack(per_layer_runs[l]["val_scwa"])
            if scwa_stack is None:
                continue
            mean_scwa, sem_scwa = mean_and_sem(scwa_stack)
            epochs = np.arange(len(mean_scwa))
            plt.plot(epochs, mean_scwa, color=colors[l], label=f"L{l}")
            plt.fill_between(
                epochs,
                mean_scwa - sem_scwa,
                mean_scwa + sem_scwa,
                color=colors[l],
                alpha=0.2,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Validation SCWA")
        plt.title(f"{ds_name}: Mean±SEM Validation SCWA")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_val_scwa_mean_sem.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated SCWA plot: {e}")
        plt.close()

    # ---------------------------------------------------- Plot 3: Best Val SCWA summary (bar, error bars)
    try:
        means = []
        sems = []
        test_means = []
        for l in sorted_layers:
            best_vals = np.array(per_layer_runs[l]["best_val"], dtype=float)
            means.append(np.nanmean(best_vals))
            sems.append(
                np.nanstd(best_vals, ddof=1) / np.sqrt(np.sum(~np.isnan(best_vals)))
            )
            # test scwa only recorded for layer that was best in each run; ignore NaNs
            test_vals = np.array(per_layer_runs[l]["test"], dtype=float)
            test_means.append(np.nanmean(test_vals))
        x = np.arange(len(sorted_layers))
        plt.figure()
        plt.bar(
            x,
            means,
            yerr=sems,
            color=[colors[l] for l in sorted_layers],
            alpha=0.7,
            capsize=5,
            label="Best Val SCWA (mean±SEM)",
        )
        # overlay test means where they exist
        for xi, t in enumerate(test_means):
            if not np.isnan(t):
                plt.bar(
                    xi,
                    t,
                    color="k",
                    alpha=0.4,
                    label="Mean Test SCWA" if xi == 0 else "",
                )
        plt.xticks(x, [f"L{l}" for l in sorted_layers])
        plt.ylabel("SCWA")
        plt.title(f"{ds_name}: Best Validation SCWA per Depth (aggregated over runs)")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_best_scwa_aggregated.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated summary bar plot: {e}")
        plt.close()
