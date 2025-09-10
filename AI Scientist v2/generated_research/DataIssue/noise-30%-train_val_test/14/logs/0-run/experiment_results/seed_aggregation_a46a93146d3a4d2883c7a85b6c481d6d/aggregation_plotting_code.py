import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")  # in case we are on a head-less machine

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# 1) Load every experiment_data.npy that is listed
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-22_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_3fde43d6cf81461e9b957d61b8627901_proc_3462555/experiment_data.npy",
    "experiments/2025-08-17_23-44-22_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_ec6b0cc0444648c0972a6bf53c1379fb_proc_3462553/experiment_data.npy",
    "experiments/2025-08-17_23-44-22_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_d84eec3083be415caa30b7232c125db5_proc_3462554/experiment_data.npy",
]

all_experiment_data = []
for rel_path in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"[WARN] Could not load {rel_path}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded; aborting plotting.")
    quit()

# ------------------------------------------------------------
dataset_name = "SPR_BENCH"
section_name = "d_model_tuning"

# Aggregation containers: dict[d_model] -> dict[epoch] -> list(values_across_runs)
metric_collectors = {
    "val_f1": defaultdict(lambda: defaultdict(list)),
    "val_loss": defaultdict(lambda: defaultdict(list)),
}

final_val_f1 = defaultdict(list)  # d_model -> list of final f1 for each run

for exp in all_experiment_data:
    section = exp.get(section_name, {})
    for d_model, ds_dict in section.items():
        if dataset_name not in ds_dict:
            continue
        run = ds_dict[dataset_name]
        val_metrics = run["metrics"]["val"]
        val_losses = run["losses"]["val"]
        # assume len(val_metrics) == len(val_losses)
        for m, l in zip(val_metrics, val_losses):
            epoch = m["epoch"]
            metric_collectors["val_f1"][d_model][epoch].append(m["macro_f1"])
            metric_collectors["val_loss"][d_model][epoch].append(l["loss"])
        # final value for bar chart
        if val_metrics:
            final_val_f1[d_model].append(val_metrics[-1]["macro_f1"])


# helper to compute mean and stderr arrays per d_model
def build_curve(data_dict):
    epochs_sorted = sorted({e for d in data_dict.values() for e in d.keys()})
    curve = {}
    for d_model, epoch_dict in data_dict.items():
        xs, means, errs = [], [], []
        for e in epochs_sorted:
            vals = epoch_dict.get(e, [])
            if not vals:
                continue
            xs.append(e)
            arr = np.asarray(vals, dtype=float)
            means.append(arr.mean())
            errs.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
        curve[d_model] = (np.asarray(xs), np.asarray(means), np.asarray(errs))
    return curve


curves_f1 = build_curve(metric_collectors["val_f1"])
curves_loss = build_curve(metric_collectors["val_loss"])

# ------------------------------------------------------------
# 2) Plotting
# NOTE: each figure in its own try/except

# (a) Validation Macro-F1 curves (mean ± SE)
try:
    plt.figure()
    for d_model, (xs, m, se) in curves_f1.items():
        if len(xs) > 200:  # subsample for readability
            idx = np.linspace(0, len(xs) - 1, 200, dtype=int)
            xs, m, se = xs[idx], m[idx], se[idx]
        plt.plot(xs, m, label=f"{d_model} mean")
        plt.fill_between(xs, m - se, m + se, alpha=0.3, label=f"{d_model} ±SE")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Macro-F1")
    plt.title("Validation Macro-F1 (mean ± SE)\nDataset: SPR_BENCH")
    plt.legend()
    save_name = os.path.join(working_dir, "SPR_BENCH_val_f1_mean_se.png")
    plt.savefig(save_name)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated F1 curves: {e}")
    plt.close()

# (b) Validation Loss curves (mean ± SE)
try:
    plt.figure()
    for d_model, (xs, m, se) in curves_loss.items():
        if len(xs) > 200:
            idx = np.linspace(0, len(xs) - 1, 200, dtype=int)
            xs, m, se = xs[idx], m[idx], se[idx]
        plt.plot(xs, m, label=f"{d_model} mean")
        plt.fill_between(xs, m - se, m + se, alpha=0.3, label=f"{d_model} ±SE")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss (mean ± SE)\nDataset: SPR_BENCH")
    plt.legend()
    save_name = os.path.join(working_dir, "SPR_BENCH_val_loss_mean_se.png")
    plt.savefig(save_name)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# (c) Bar chart of final validation Macro-F1
try:
    plt.figure()
    d_models = sorted(final_val_f1.keys(), key=int)
    means = [np.mean(final_val_f1[d]) for d in d_models]
    ses = [
        (
            np.std(final_val_f1[d], ddof=1) / np.sqrt(len(final_val_f1[d]))
            if len(final_val_f1[d]) > 1
            else 0.0
        )
        for d in d_models
    ]
    xs = np.arange(len(d_models))
    plt.bar(xs, means, yerr=ses, capsize=4)
    plt.xticks(xs, d_models)
    plt.xlabel("d_model")
    plt.ylabel("Final Validation Macro-F1 (mean ± SE)")
    plt.title("Final Val Macro-F1 per d_model\nDataset: SPR_BENCH")
    save_name = os.path.join(working_dir, "SPR_BENCH_final_val_f1_mean_se_bar.png")
    plt.savefig(save_name)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated bar chart: {e}")
    plt.close()

# ------------------------------------------------------------
# 3) Print numeric table
print("\nAggregated Final Validation Macro-F1 (mean ± SE) for SPR_BENCH")
for d_model in sorted(final_val_f1.keys(), key=int):
    arr = np.asarray(final_val_f1[d_model], dtype=float)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    print(f"d_model {d_model:>4}: {mean:.4f} ± {se:.4f}")
