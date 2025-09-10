import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------
# basic set-up
# -------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------
# load all experiment_data files that were supplied
# -------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_eca2e524b5c34ba391a176baea24e0bf_proc_3160641/experiment_data.npy",
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_1195467d8fb944a799d6f096b897d8d1_proc_3160639/experiment_data.npy",
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_18f10014191c46559582eb4fe8ed013c_proc_3160638/experiment_data.npy",
]
all_experiment_data = []
try:
    for path in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
    print(f"Loaded {len(all_experiment_data)} experiment files")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# -------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------
def align_and_stack(arr_list):
    """
    Trim every 1-D array in arr_list to the minimum length, then stack.
    Returns ndarray of shape [n_runs, min_len]
    """
    if not arr_list:
        return None
    min_len = min(len(x) for x in arr_list)
    arr_list = [np.asarray(x)[:min_len] for x in arr_list]
    return np.stack(arr_list, axis=0)  # [runs, epochs]


def mean_sem(arr, axis=0):
    """
    Given ndarray arr, returns mean and standard error along axis.
    """
    if arr is None or arr.size == 0:
        return None, None
    mean = arr.mean(axis=axis)
    sem = arr.std(axis=axis, ddof=1) / np.sqrt(arr.shape[axis])
    return mean, sem


# -------------------------------------------------------------
# Aggregate data across runs
# -------------------------------------------------------------
# gather every dropout tag observed in any run
all_dropout_tags = []
for exp in all_experiment_data:
    all_dropout_tags.extend(list(exp.get("dropout_tuning", {}).keys()))
all_dropout_tags = sorted(list(set(all_dropout_tags)))[:5]  # keep at most five

aggregated = {}  # tag -> dict of aggregated arrays
for tag in all_dropout_tags:
    train_loss_runs, val_loss_runs = [], []
    train_f1_runs, val_f1_runs = [], []
    test_f1_runs = []
    for exp in all_experiment_data:
        d = exp.get("dropout_tuning", {}).get(tag, None)
        if d is None:
            continue
        train_loss_runs.append(np.asarray(d["losses"]["train"]))
        val_loss_runs.append(np.asarray(d["losses"]["val"]))
        train_f1_runs.append(np.asarray(d["metrics"]["train_f1"]))
        val_f1_runs.append(np.asarray(d["metrics"]["val_f1"]))
        test_f1_runs.append(float(d["metrics"]["test_f1"]))
    aggregated[tag] = {
        "train_loss": align_and_stack(train_loss_runs),
        "val_loss": align_and_stack(val_loss_runs),
        "train_f1": align_and_stack(train_f1_runs),
        "val_f1": align_and_stack(val_f1_runs),
        "test_f1": np.asarray(test_f1_runs) if test_f1_runs else None,
    }

# -------------------------------------------------------------
# 1) aggregated loss curves (mean ± SEM)
# -------------------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # Left ‑ training loss
    plt.subplot(1, 2, 1)
    for tag in aggregated:
        arr = aggregated[tag]["train_loss"]
        if arr is None:
            continue
        mu, se = mean_sem(arr)
        epochs = np.arange(len(mu))
        plt.plot(epochs, mu, label=f"{tag} mean")
        plt.fill_between(epochs, mu - se, mu + se, alpha=0.2)
    plt.title("Left: Training Loss (mean ± SEM) - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=6)

    # Right ‑ validation loss
    plt.subplot(1, 2, 2)
    for tag in aggregated:
        arr = aggregated[tag]["val_loss"]
        if arr is None:
            continue
        mu, se = mean_sem(arr)
        epochs = np.arange(len(mu))
        plt.plot(epochs, mu, label=f"{tag} mean")
        plt.fill_between(epochs, mu - se, mu + se, alpha=0.2)
    plt.title("Right: Validation Loss (mean ± SEM) - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=6)

    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_mean_sem.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# -------------------------------------------------------------
# 2) aggregated F1 curves (mean ± SEM)
# -------------------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # Left – training F1
    plt.subplot(1, 2, 1)
    for tag in aggregated:
        arr = aggregated[tag]["train_f1"]
        if arr is None:
            continue
        mu, se = mean_sem(arr)
        epochs = np.arange(len(mu))
        plt.plot(epochs, mu, label=f"{tag} mean")
        plt.fill_between(epochs, mu - se, mu + se, alpha=0.2)
    plt.title("Left: Training Macro-F1 (mean ± SEM) - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend(fontsize=6)

    # Right – validation F1
    plt.subplot(1, 2, 2)
    for tag in aggregated:
        arr = aggregated[tag]["val_f1"]
        if arr is None:
            continue
        mu, se = mean_sem(arr)
        epochs = np.arange(len(mu))
        plt.plot(epochs, mu, label=f"{tag} mean")
        plt.fill_between(epochs, mu - se, mu + se, alpha=0.2)
    plt.title("Right: Validation Macro-F1 (mean ± SEM) - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend(fontsize=6)

    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves_mean_sem.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated F1 curves: {e}")
    plt.close()

# -------------------------------------------------------------
# 3) Test-set Macro-F1 bar plot (mean ± SEM)
# -------------------------------------------------------------
try:
    plt.figure()
    tags = list(aggregated.keys())
    means = []
    sems = []
    xtick_lbls = []
    for tag in tags:
        tf1 = aggregated[tag]["test_f1"]
        if tf1 is None or tf1.size == 0:
            means.append(np.nan)
            sems.append(0)
        else:
            means.append(tf1.mean())
            sems.append(tf1.std(ddof=1) / np.sqrt(len(tf1)))
        xtick_lbls.append(tag.split("_")[-1])  # show only dropout value
    x = np.arange(len(tags))
    plt.bar(x, means, yerr=sems, capsize=4)
    plt.title("Test Macro-F1 by Dropout (mean ± SEM) - SPR_BENCH")
    plt.xlabel("Dropout")
    plt.ylabel("Macro-F1")
    plt.xticks(x, xtick_lbls)
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar_mean_sem.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated test-F1 bar: {e}")
    plt.close()

# -------------------------------------------------------------
# Print aggregated scalar results to console
# -------------------------------------------------------------
print("Aggregated Test Macro-F1 (mean ± SEM):")
for tag in aggregated:
    tf1 = aggregated[tag]["test_f1"]
    if tf1 is None or tf1.size == 0:
        continue
    mu = tf1.mean()
    se = tf1.std(ddof=1) / np.sqrt(len(tf1))
    print(f"  {tag}: {mu:.4f} ± {se:.4f}")
