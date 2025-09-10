import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------
# 1. Load every experiment_data.npy that the orchestrator produced
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-22_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_3e2bb0f24be44712890754709df90091_proc_3469038/experiment_data.npy",
    "experiments/2025-08-17_23-44-22_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_10adfd972499486080b7d3a2eb3036c8_proc_3469040/experiment_data.npy",
    "experiments/2025-08-17_23-44-22_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_404b3ac67c35477aac307b3a698caf87_proc_3469037/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

# --------------------------------------------------------------
# 2. Aggregate runs -> {dataset: {model: [run_dict, ...]}}
aggregated = {}
for exp in all_experiment_data:
    for dset, runs in exp.items():
        aggregated.setdefault(dset, {})
        for model, run_dict in runs.items():
            aggregated[dset].setdefault(model, []).append(run_dict)


# --------------------------------------------------------------
def stack_metric(run_list, split, metric_key):
    """
    Build (epochs, stacked_values) where
    - epochs is a list of common epochs across runs ordered by appearance
    - stacked_values is a dict epoch -> list(values)
    """
    common_epochs = None
    epoch_to_vals = {}
    # First, find epochs common to all runs
    for r in run_list:
        e_list = [e["epoch"] for e in r["metrics"][split]]
        common_epochs = (
            set(e_list) if common_epochs is None else common_epochs.intersection(e_list)
        )
    common_epochs = sorted(list(common_epochs))
    # Collect metric values
    for e in common_epochs:
        epoch_to_vals[e] = []
        for r in run_list:
            # find metric for epoch e
            for m in r["metrics"][split]:
                if m["epoch"] == e:
                    epoch_to_vals[e].append(m[metric_key])
                    break
    return common_epochs, epoch_to_vals


# helper to convert dict epoch->list(values) to sorted arrays
def mean_se_arrays(epoch_to_vals):
    epochs = sorted(epoch_to_vals.keys())
    means = np.array([np.mean(epoch_to_vals[e]) for e in epochs])
    ses = np.array(
        [
            (
                np.std(epoch_to_vals[e], ddof=1) / np.sqrt(len(epoch_to_vals[e]))
                if len(epoch_to_vals[e]) > 1
                else 0.0
            )
            for e in epochs
        ]
    )
    return epochs, means, ses


# --------------------------------------------------------------
for dataset, model_runs in aggregated.items():
    # We will keep numerical summary for console print
    summary = {}
    # ------------- LOSS CURVES ----------------------
    try:
        plt.figure()
        for model, runs in model_runs.items():
            # Train
            ep_train, d_train = stack_metric(runs, "train", "loss")
            ep_val, d_val = stack_metric(runs, "val", "loss")
            if not ep_train or not ep_val:
                continue
            ep_t, m_t, se_t = mean_se_arrays(d_train)
            ep_v, m_v, se_v = mean_se_arrays(d_val)
            plt.plot(ep_t, m_t, "--", label=f"{model}-train μ")
            plt.fill_between(ep_t, m_t - se_t, m_t + se_t, alpha=0.2)
            plt.plot(ep_v, m_v, label=f"{model}-val μ")
            plt.fill_between(ep_v, m_v - se_v, m_v + se_v, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"Mean ± SE Training vs Validation Loss\nDataset: {dataset}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_loss_mean_se.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves for {dataset}: {e}")
        plt.close()

    # ------------- F1 CURVES ------------------------
    try:
        plt.figure()
        for model, runs in model_runs.items():
            ep_train, d_train = stack_metric(runs, "train", "macro_f1")
            ep_val, d_val = stack_metric(runs, "val", "macro_f1")
            if not ep_train or not ep_val:
                continue
            ep_t, m_t, se_t = mean_se_arrays(d_train)
            ep_v, m_v, se_v = mean_se_arrays(d_val)
            plt.plot(ep_t, m_t, "--", label=f"{model}-train μ")
            plt.fill_between(ep_t, m_t - se_t, m_t + se_t, alpha=0.2)
            plt.plot(ep_v, m_v, label=f"{model}-val μ")
            plt.fill_between(ep_v, m_v - se_v, m_v + se_v, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"Mean ± SE Training vs Validation Macro-F1\nDataset: {dataset}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_f1_mean_se.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated F1 curves for {dataset}: {e}")
        plt.close()

    # ------------- FINAL VAL F1 BAR -----------------
    try:
        plt.figure()
        models = []
        means = []
        ses = []
        for model, runs in model_runs.items():
            vals = []
            for r in runs:
                if r["metrics"]["val"]:
                    vals.append(r["metrics"]["val"][-1]["macro_f1"])
            if not vals:
                continue
            models.append(model)
            means.append(np.mean(vals))
            ses.append(
                np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            )
            summary[model] = (np.mean(vals), ses[-1])
        x = np.arange(len(models))
        plt.bar(x, means, yerr=ses, capsize=4)
        plt.xticks(x, models, rotation=45, ha="right")
        plt.ylabel("Final Validation Macro-F1")
        plt.xlabel("Model")
        plt.title(f"Final Val Macro-F1 (Mean ± SE)\nDataset: {dataset}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset}_final_val_f1_mean_se_bar.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated final F1 bar chart for {dataset}: {e}")
        plt.close()

    # Console summary
    if summary:
        print(f"=== {dataset} Final Validation Macro-F1 ===")
        for m, (mu, se) in summary.items():
            print(f"{m:20s}: {mu:.4f} ± {se:.4f}")
