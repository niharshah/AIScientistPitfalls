import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load all experiment_data dicts ---------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_573920eca39f485bab1a49bfd7816f2f_proc_3158727/experiment_data.npy",
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_3d431d4d26dc4023824f7680e9e21620_proc_3158728/experiment_data.npy",
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_d7a04eeff3f6467493663d873a6d0e39_proc_3158726/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment results could be loaded; aborting plotting.")
    exit()

# ------------------------------------------------------------------
# 2. Aggregate metrics ---------------------------------------------
# Structure: agg_data[key] = list_of_run_dicts_for_that_key
agg_data = {}
for run in all_experiment_data:
    for k, v in run.get("num_epochs", {}).items():
        agg_data.setdefault(k, []).append(v)


def _stack_and_stats(list_of_arrays):
    """Returns (mean, se) along axis 0."""
    arr = np.stack(list_of_arrays, axis=0)  # shape [n_runs, time]
    mean = np.mean(arr, axis=0)
    se = (
        np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, se


# Colours for distinct hyper-param keys
keys = list(agg_data.keys())
colors = plt.cm.tab10.colors if keys else []

# ------------------------------------------------------------------
# 3. Train/Val Macro-F1 curves with SE shading ----------------------
try:
    plt.figure()
    for idx, k in enumerate(keys):
        runs_for_k = agg_data[k]
        # assume all epochs arrays are identical – take from first run
        epochs = runs_for_k[0]["epochs"]
        tr_arrays = [
            r["metrics"]["train_macro_f1"]
            for r in runs_for_k
            if "metrics" in r and "train_macro_f1" in r["metrics"]
        ]
        val_arrays = [
            r["metrics"]["val_macro_f1"]
            for r in runs_for_k
            if "metrics" in r and "val_macro_f1" in r["metrics"]
        ]
        if not tr_arrays or not val_arrays:
            continue
        tr_mean, tr_se = _stack_and_stats(tr_arrays)
        val_mean, val_se = _stack_and_stats(val_arrays)
        c = colors[idx % len(colors)]
        # Train
        plt.plot(epochs, tr_mean, linestyle="--", color=c, label=f"{k}-train mean")
        plt.fill_between(epochs, tr_mean - tr_se, tr_mean + tr_se, color=c, alpha=0.2)
        # Val
        plt.plot(epochs, val_mean, linestyle="-", color=c, label=f"{k}-val mean")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, color=c, alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(
        "SPR_BENCH Aggregate Macro-F1 (shaded = ±SE)\nLeft dashed: Train, Right solid: Validation"
    )
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_macro_f1_curves_agg.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated macro-F1 plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4. Train/Val Loss curves with SE shading --------------------------
try:
    plt.figure()
    for idx, k in enumerate(keys):
        runs_for_k = agg_data[k]
        epochs = runs_for_k[0]["epochs"]
        tr_arrays = [
            r["losses"]["train"]
            for r in runs_for_k
            if "losses" in r and "train" in r["losses"]
        ]
        val_arrays = [
            r["losses"]["val"]
            for r in runs_for_k
            if "losses" in r and "val" in r["losses"]
        ]
        if not tr_arrays or not val_arrays:
            continue
        tr_mean, tr_se = _stack_and_stats(tr_arrays)
        val_mean, val_se = _stack_and_stats(val_arrays)
        c = colors[idx % len(colors)]
        plt.plot(epochs, tr_mean, linestyle="--", color=c, label=f"{k}-train mean")
        plt.fill_between(epochs, tr_mean - tr_se, tr_mean + tr_se, color=c, alpha=0.2)
        plt.plot(epochs, val_mean, linestyle="-", color=c, label=f"{k}-val mean")
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, color=c, alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "SPR_BENCH Aggregate Loss (shaded = ±SE)\nLeft dashed: Train, Right solid: Validation"
    )
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves_agg.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 5. Test Macro-F1 bar chart (mean ± SE) ----------------------------
aggregated_test_scores = {}
try:
    # collect arrays of final test Macro-F1 per key
    for k in keys:
        values = []
        for run_dict in agg_data[k]:
            if "test_macro_f1" in run_dict:
                values.append(run_dict["test_macro_f1"])
        if values:
            values = np.asarray(values)
            aggregated_test_scores[k] = (
                values.mean(),
                values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0,
            )

    if aggregated_test_scores:
        plt.figure()
        means = [v[0] for v in aggregated_test_scores.values()]
        ses = [v[1] for v in aggregated_test_scores.values()]
        plt.bar(
            range(len(means)),
            means,
            yerr=ses,
            capsize=5,
            tick_label=list(aggregated_test_scores.keys()),
        )
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Aggregate Test Macro-F1 (bars = mean, error = ±SE)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_test_macro_f1_bar_agg.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating aggregated test score bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 6. Numeric summary -----------------------------------------------
print("Aggregated Test Macro-F1 (mean, ±SE):")
for k, (m, se) in aggregated_test_scores.items():
    print(f"  {k}: {m:.4f} ± {se:.4f}")
