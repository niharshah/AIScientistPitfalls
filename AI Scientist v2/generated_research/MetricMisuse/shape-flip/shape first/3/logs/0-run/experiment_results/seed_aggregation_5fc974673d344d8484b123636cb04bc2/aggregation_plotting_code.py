import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load multiple experiment files ----------
experiment_data_path_list = [
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_48fe5d59d19b4eec8de94f1da6aff3ea_proc_456840/experiment_data.npy",
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_d16d35eac9af417b87d9385a75f5ace7_proc_456839/experiment_data.npy",
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_6352f3a1515f43c88021df6a6828adf5_proc_456837/experiment_data.npy",
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


# helper
def get(data, *keys, default=None):
    for k in keys:
        data = data.get(k, {})
    return data if data != {} else default


ds_name = "SPR_BENCH"
runs = [exp.get(ds_name, {}) for exp in all_experiment_data if ds_name in exp]

if not runs:
    print(f"No runs found for dataset {ds_name}")
    exit()


# ---------- aggregate per-epoch arrays ----------
def stack_with_nan(list_of_lists):
    max_len = max(len(lst) for lst in list_of_lists)
    out = []
    for lst in list_of_lists:
        padded = np.array(lst + [np.nan] * (max_len - len(lst)), dtype=float)
        out.append(padded)
    return np.vstack(out)


loss_train_mat = stack_with_nan([get(r, "losses", "train", default=[]) for r in runs])
loss_val_mat = stack_with_nan([get(r, "losses", "val", default=[]) for r in runs])

metrics_keys = ["SWA", "CWA", "HWA"]
metric_mats = {
    k: stack_with_nan(
        [[d.get(k, np.nan) for d in get(r, "metrics", "val", default=[])] for r in runs]
    )
    for k in metrics_keys
}

epochs = np.arange(1, loss_train_mat.shape[1] + 1)


# convenience for mean & sem
def mean_sem(mat):
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return mean, sem


# ---------- Plot 1: aggregated loss curves ----------
try:
    plt.figure()
    for mat, lbl, col in [
        (loss_train_mat, "Train Loss", "tab:blue"),
        (loss_val_mat, "Val Loss", "tab:orange"),
    ]:
        mean, sem = mean_sem(mat)
        plt.plot(epochs, mean, label=lbl, color=col)
        plt.fill_between(epochs, mean - sem, mean + sem, color=col, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_name}: Mean ± SEM Loss over {len(runs)} runs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ---------- Plot 2: aggregated validation metrics ----------
try:
    plt.figure()
    colors = {"SWA": "tab:green", "CWA": "tab:red", "HWA": "tab:pink"}
    for k in metrics_keys:
        mean, sem = mean_sem(metric_mats[k])
        plt.plot(epochs, mean, label=f"{k} mean", color=colors[k])
        plt.fill_between(epochs, mean - sem, mean + sem, color=colors[k], alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title(f"{ds_name}: Validation metrics (mean ± SEM, {len(runs)} runs)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_val_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metric plot: {e}")
    plt.close()

# ---------- Plot 3: final epoch bar chart with error bars ----------
try:
    final_vals = {k: metric_mats[k][:, -1] for k in metrics_keys}
    means = [np.nanmean(final_vals[k]) for k in metrics_keys]
    sems = [
        np.nanstd(final_vals[k]) / np.sqrt(np.sum(~np.isnan(final_vals[k])))
        for k in metrics_keys
    ]
    plt.figure()
    x = np.arange(len(metrics_keys))
    plt.bar(x, means, yerr=sems, capsize=5, color=["skyblue", "lightgreen", "salmon"])
    plt.xticks(x, metrics_keys)
    plt.ylim(0, 1)
    for i, (m, s) in enumerate(zip(means, sems)):
        plt.text(i, m + 0.02, f"{m:.2f}±{s:.2f}", ha="center")
    plt.title(f"{ds_name}: Final validation metrics (mean ± SEM, {len(runs)} runs)")
    plt.savefig(os.path.join(working_dir, f"{ds_name}_agg_final_metrics_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final metric bar plot: {e}")
    plt.close()

# ---------- Plot 4: confusion matrix of first run (illustrative) ----------
try:
    y_true_t = runs[0].get("ground_truth", [])
    y_pred_t = runs[0].get("predictions", [])
    if y_true_t and y_pred_t:
        labels = sorted(list(set(y_true_t) | set(y_pred_t)))
        idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), int)
        for yt, yp in zip(y_true_t, y_pred_t):
            cm[idx[yt], idx[yp]] += 1
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_name}: Confusion Matrix (example run)")
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_confusion_matrix_example.png"),
            bbox_inches="tight",
        )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print final averaged metrics ----------
print(f"{ds_name} final epoch mean metrics across {len(runs)} runs:")
for k, m, s in zip(metrics_keys, means, sems):
    print(f"  {k}: {m:.4f} ± {s:.4f}")
