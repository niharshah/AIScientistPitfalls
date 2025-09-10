import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------- #
# Setup & load data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- #
# List of experiment result files (relative to $AI_SCIENTIST_ROOT)
experiment_data_path_list = [
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_d88cc6ba07294bf19dfd3113e3a84a69_proc_3039480/experiment_data.npy",
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_7fd20bec3327401a812995812cb62863_proc_3039482/experiment_data.npy",
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_3c5d711f32d94e318a8bdd81f7382098_proc_3039483/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_p, allow_pickle=True).item()
        # Keep only the relevant sub-dict
        if "multi_synth_generalization" in exp:
            all_experiment_data.append(exp["multi_synth_generalization"])
except Exception as e:
    print(f"Error loading experiment data: {e}")

if len(all_experiment_data) == 0:
    print("No experiment data could be loaded – aborting.")
else:
    # Determine datasets that exist in every run
    datasets = ["variety", "length_parity", "majority_shape"]
    n_runs = len(all_experiment_data)

    # --------------------------------------------------------------- #
    # 1-3) Dataset-specific loss curves (mean ± stderr)
    for ds in datasets:
        try:
            # Collect curves
            tr_mat, val_mat = [], []
            for exp in all_experiment_data:
                if ds not in exp:
                    continue
                tr_mat.append(exp[ds]["losses"]["train"])
                val_mat.append(exp[ds]["losses"]["val"])
            tr_mat = np.array(tr_mat)
            val_mat = np.array(val_mat)
            if tr_mat.size == 0 or val_mat.size == 0:
                raise ValueError(f"No data for {ds}")

            mean_tr, se_tr = tr_mat.mean(0), tr_mat.std(0) / np.sqrt(tr_mat.shape[0])
            mean_val, se_val = val_mat.mean(0), val_mat.std(0) / np.sqrt(
                val_mat.shape[0]
            )
            epochs = np.arange(1, len(mean_tr) + 1)

            plt.figure()
            plt.plot(epochs, mean_tr, label="Train Loss (mean)", color="tab:blue")
            plt.fill_between(
                epochs, mean_tr - se_tr, mean_tr + se_tr, alpha=0.3, color="tab:blue"
            )
            plt.plot(
                epochs, mean_val, label="Validation Loss (mean)", color="tab:orange"
            )
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                alpha=0.3,
                color="tab:orange",
            )
            plt.title(f"{ds} dataset – Training vs Validation Loss (mean ± SE)")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_loss_curves_mean_se.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds}: {e}")
            plt.close()

    # --------------------------------------------------------------- #
    # 4) Aggregate validation-accuracy curves across datasets
    try:
        plt.figure()
        for ds in datasets:
            val_mat = []
            for exp in all_experiment_data:
                if ds in exp:
                    val_mat.append(exp[ds]["metrics"]["val"])
            val_mat = np.array(val_mat)
            if val_mat.size == 0:
                continue
            mean_val = val_mat.mean(0)
            se_val = val_mat.std(0) / np.sqrt(val_mat.shape[0])
            epochs = np.arange(1, len(mean_val) + 1)
            plt.plot(epochs, mean_val, label=f"{ds} (mean)")
            plt.fill_between(epochs, mean_val - se_val, mean_val + se_val, alpha=0.25)
        plt.title("Validation Accuracy Across Datasets (mean ± SE)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        fname = os.path.join(working_dir, "val_accuracy_all_datasets_mean_se.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot: {e}")
        plt.close()

    # --------------------------------------------------------------- #
    # 5) Transfer accuracy heat-map (mean across runs)
    try:
        n_ds = len(datasets)
        mat_runs = np.zeros((n_runs, n_ds, n_ds))
        for run_idx, exp in enumerate(all_experiment_data):
            for i, src in enumerate(datasets):
                # diagonal: final val acc on the same dataset
                if src in exp:
                    mat_runs[run_idx, i, i] = exp[src]["metrics"]["val"][-1]
                for j, tgt in enumerate(datasets):
                    if i == j:
                        continue
                    key = f"{src}_to_{tgt}"
                    if "transfer" in exp and key in exp["transfer"]:
                        mat_runs[run_idx, i, j] = exp["transfer"][key]["acc"]
        mat_mean = mat_runs.mean(0)
        mat_se = mat_runs.std(0) / np.sqrt(n_runs)

        plt.figure()
        im = plt.imshow(mat_mean, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(im)
        plt.xticks(range(n_ds), datasets, rotation=45)
        plt.yticks(range(n_ds), datasets)
        plt.title(
            "Transfer Accuracy Heat-map (mean values)\nLeft/Top: Source, Right/Bottom: Target"
        )
        # annotate with mean±SE
        for i in range(n_ds):
            for j in range(n_ds):
                txt = f"{mat_mean[i, j]:.2f}"
                plt.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color="w" if mat_mean[i, j] < 0.5 else "k",
                )
        fname = os.path.join(working_dir, "transfer_accuracy_heatmap_mean.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating transfer heat-map: {e}")
        plt.close()

    # --------------------------------------------------------------- #
    # Console summary of final validation accuracies
    try:
        final_stats = {}
        for ds in datasets:
            finals = []
            for exp in all_experiment_data:
                if ds in exp:
                    finals.append(exp[ds]["metrics"]["val"][-1])
            finals = np.array(finals)
            if finals.size > 0:
                final_stats[ds] = (finals.mean(), finals.std())
        print("Final Validation Accuracies (mean ± std):")
        for ds, (m, s) in final_stats.items():
            print(f"  {ds}: {m:.3f} ± {s:.3f}")
    except Exception as e:
        print(f"Error printing final stats: {e}")
