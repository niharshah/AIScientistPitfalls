import matplotlib.pyplot as plt
import numpy as np
import os
import math

# --------------------------------------------------
# directory where plots will be written
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------
# paths to experiment_data.npy files  (provided)
experiment_data_path_list = [
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_ca5aeb2bde954c7990cbfb51d20e6c44_proc_2602195/experiment_data.npy",
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_67a49514085949aa97986b16d4273a62_proc_2602196/experiment_data.npy",
    "experiments/2025-08-14_01-55-43_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_ebb58693b9ce486187cf37de9fc87d7d_proc_2602193/experiment_data.npy",
]

# --------------------------------------------------
# LOAD ALL RUNS
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        d = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(d)
except Exception as e:
    print(f"Error loading experiment data: {e}")

if not all_experiment_data:
    print("No experiment data available – exiting")
    exit()

# --------------------------------------------------
# discover common dataset key (assume one per run)
dataset_name = list(all_experiment_data[0].keys())[0]


# helper to compute mean & sem stacked along new axis 0
def mean_sem(stack):
    stack = np.asarray(stack)
    mean = np.nanmean(stack, axis=0)
    sem = np.nanstd(stack, axis=0) / math.sqrt(stack.shape[0])
    return mean, sem


# --------------------------------------------------
# --------- 1) AGGREGATE EPOCH-LEVEL CURVES ----------
epoch_metrics_to_look_for = [
    "loss",
    "accuracy",
    "acc",
    "f1",
    "val_loss",
    "val_accuracy",
    "val_acc",
]
# map metric_name -> {'train': [runs], 'val': [runs]}
aggregated_curves = {}

for run_data in all_experiment_data:
    metrics = run_data.get(dataset_name, {}).get("metrics", {})
    if not metrics:
        continue
    for split, split_dict in metrics.items():  # e.g. 'train', 'val'
        for mname, values in split_dict.items():
            if mname not in epoch_metrics_to_look_for:
                continue
            key = f"{split}_{mname}"
            aggregated_curves.setdefault(key, []).append(np.asarray(values))

# trim curves to same length
for k, runs in aggregated_curves.items():
    min_len = min([len(r) for r in runs])
    aggregated_curves[k] = [r[:min_len] for r in runs]

# PLOT each aggregated curve
for k, runs in aggregated_curves.items():
    try:
        mean, sem = mean_sem(runs)
        epochs = np.arange(1, len(mean) + 1)

        plt.figure()
        plt.plot(epochs, mean, label=f"Mean {k}")
        plt.fill_between(epochs, mean - sem, mean + sem, alpha=0.3, label="±1 SEM")
        plt.xlabel("Epoch")
        plt.ylabel(k)
        plt.title(f"{dataset_name}: {k} vs Epoch\nMean ± SEM across {len(runs)} runs")
        plt.legend()
        fname = f"{dataset_name}_{k}_curve.png".replace("/", "_")
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {k}: {e}")
        plt.close()

# --------------------------------------------------
# --------- 2) AGGREGATE FINAL TEST METRICS ----------
scalar_metrics_to_collect = ["ACC", "SWA", "CWA", "NRGS", "accuracy", "f1"]
collected_scalars = {m: [] for m in scalar_metrics_to_collect}

for run_data in all_experiment_data:
    test_metrics = run_data.get(dataset_name, {}).get("metrics", {}).get("test", {})
    for m, v in test_metrics.items():
        if m in collected_scalars:
            collected_scalars[m].append(float(v))

# remove empty entries
collected_scalars = {k: v for k, v in collected_scalars.items() if v}

try:
    if collected_scalars:
        names, means, sems = [], [], []
        for m, vals in collected_scalars.items():
            names.append(m)
            mean, sem = mean_sem(vals)
            means.append(mean)
            sems.append(sem)

        x = np.arange(len(names))
        plt.figure()
        plt.bar(x, means, yerr=sems, capsize=5, alpha=0.7, label="Mean ± SEM")
        plt.xticks(x, names)
        plt.ylabel("Metric Value")
        plt.title(
            f"{dataset_name}: Final Test Metrics\nMean ± SEM across {len(all_experiment_data)} runs"
        )
        plt.legend()
        fname = f"{dataset_name}_final_test_metrics.png".replace("/", "_")
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()

        # print numerical results
        print("Aggregated test metrics (mean ± SEM):")
        for n, m, s in zip(names, means, sems):
            print(f"  {n}: {m:.4f} ± {s:.4f}")
    else:
        print("No scalar test metrics found to plot.")
except Exception as e:
    print(f"Error creating scalar metric bar chart: {e}")
    plt.close()
