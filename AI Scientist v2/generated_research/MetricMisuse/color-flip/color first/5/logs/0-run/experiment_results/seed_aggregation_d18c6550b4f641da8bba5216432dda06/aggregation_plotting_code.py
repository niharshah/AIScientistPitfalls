import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# List of relative experiment_data.npy files (taken from the prompt)
experiment_data_path_list = [
    "experiments/2025-08-31_03-13-33_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_fd9d7f4d6aff490286c536837de0df44_proc_1619820/experiment_data.npy",
    "experiments/2025-08-31_03-13-33_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_f17027563278453496207850a128f441_proc_1619819/experiment_data.npy",
    "None/experiment_data.npy",
]

all_experiment_data = []
for rel_path in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        data = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded. Exiting.")
    exit()

# ------------------------------------------------------------------
target_ds = "SPR_BENCH"
runs = [d for d in all_experiment_data if target_ds in d]

if not runs:
    print(f"Dataset {target_ds} not found in any run.")
    exit()


# ------------------------------------------------------------------
def stack_and_trim(list_of_arrays, trim_to_min=True):
    """Stack 1-D arrays after trimming them to common length."""
    if not list_of_arrays:
        return np.array([])

    lengths = [len(a) for a in list_of_arrays if len(a) > 0]
    if not lengths:
        return np.array([])

    L = min(lengths) if trim_to_min else max(lengths)
    trimmed = [a[:L] for a in list_of_arrays]
    return np.vstack(trimmed)  # shape = (n_runs, L)


# ------------------------------------------------------------------
# Aggregated Train/Val Loss ----------------------------------------
try:
    train_losses = stack_and_trim(
        [np.array(r[target_ds]["losses"].get("train", []), dtype=float) for r in runs]
    )
    val_losses = stack_and_trim(
        [np.array(r[target_ds]["losses"].get("val", []), dtype=float) for r in runs]
    )

    if train_losses.size and val_losses.size:
        epochs = np.arange(1, train_losses.shape[1] + 1)

        mean_train = train_losses.mean(axis=0)
        se_train = train_losses.std(axis=0, ddof=1) / np.sqrt(train_losses.shape[0])

        mean_val = val_losses.mean(axis=0)
        se_val = val_losses.std(axis=0, ddof=1) / np.sqrt(val_losses.shape[0])

        plt.figure()
        plt.plot(epochs, mean_train, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs,
            mean_train - se_train,
            mean_train + se_train,
            color="tab:blue",
            alpha=0.25,
            label="Train SE",
        )
        plt.plot(epochs, mean_val, label="Validation Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs,
            mean_val - se_val,
            mean_val + se_val,
            color="tab:orange",
            alpha=0.25,
            label="Val SE",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH Aggregated Training vs Validation Loss\n(Mean ± Standard Error)"
        )
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve_aggregated.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Loss arrays not found – skipping aggregated loss plot.")
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# Aggregated Validation Metrics ------------------------------------
try:
    metric_keys = ["acc", "cwa", "swa", "compwa"]
    metric_arrays = {k: [] for k in metric_keys}

    for r in runs:
        for k in metric_keys:
            arr = np.array([m[k] for m in r[target_ds]["metrics"]["val"]], dtype=float)
            metric_arrays[k].append(arr)

    # Use ACC length to define epochs (assuming all metrics same length)
    acc_stacked = stack_and_trim(metric_arrays["acc"])
    if acc_stacked.size:
        epochs = np.arange(1, acc_stacked.shape[1] + 1)

        plt.figure()
        for k, color in zip(
            metric_keys, ["tab:blue", "tab:green", "tab:red", "tab:purple"]
        ):
            stacked = stack_and_trim(metric_arrays[k])
            mean = stacked.mean(axis=0)
            se = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            plt.errorbar(
                epochs,
                mean,
                yerr=se,
                label=f"{k.upper()} (mean±SE)",
                marker="o",
                linestyle="-",
                color=color,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title(
            "SPR_BENCH Aggregated Validation Metrics Over Epochs\n(Mean ± Standard Error)"
        )
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_validation_metrics_aggregated.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Validation metric arrays not found – skipping aggregated metrics plot.")
except Exception as e:
    print(f"Error creating aggregated validation metrics plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print mean ± std of final Test Metrics ---------------------------
test_metric_dicts = [
    r[target_ds]["metrics"].get("test", {})
    for r in runs
    if r[target_ds]["metrics"].get("test", {})
]
if test_metric_dicts:
    # Collect by key
    keys = test_metric_dicts[0].keys()
    print("Aggregated Test Metrics (mean ± std):")
    for k in keys:
        vals = np.array([d[k] for d in test_metric_dicts], dtype=float)
        print(f"  {k}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")
else:
    print("No test metrics found across runs.")
