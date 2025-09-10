import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Basic set-up
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load every experiment_data.npy that was provided
# ------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_62063839faac483db306ecb96ecfb800_proc_3202617/experiment_data.npy",
        "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_e5ce8cad06e04133bff629302b0162d5_proc_3202616/experiment_data.npy",
        "experiments/2025-08-17_02-43-44_interpretable_neural_rule_learning_attempt_0/logs/0-run/experiment_results/experiment_9dc0ef42cddf4d12801e4b230fd859f5_proc_3202618/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if not os.path.isfile(full_path):
            print(f"Warning: {full_path} not found")
            continue
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Quit early if nothing found
if not all_experiment_data:
    print("No experiment data could be loaded – exiting.")
    exit()


# ------------------------------------------------------------------
# 2. Aggregate metrics across runs (focus on 'SPR_BENCH' dataset)
# ------------------------------------------------------------------
def pad_to_max(list_of_1d_arrays):
    max_len = max(len(a) for a in list_of_1d_arrays)
    padded = np.full((len(list_of_1d_arrays), max_len), np.nan, dtype=float)
    for i, a in enumerate(list_of_1d_arrays):
        padded[i, : len(a)] = a
    return padded


agg = {}  # {hidden_dim: {"train_acc":2d, "val_acc":2d, "train_loss":2d, "val_loss":2d}}
best_test_acc_all, best_fid_all = [], []

for exp in all_experiment_data:
    data_key = exp.get("hidden_dim", {}).get("SPR_BENCH", {})
    metrics_all = data_key.get("metrics", {})
    losses_all = data_key.get("losses", {})
    for hd, m in metrics_all.items():
        if not isinstance(hd, int):
            continue
        entry = agg.setdefault(
            hd, {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
        )
        entry["train_acc"].append(np.asarray(m["train_acc"], dtype=float))
        entry["val_acc"].append(np.asarray(m["val_acc"], dtype=float))
        # losses may be saved separately
        entry["val_loss"].append(np.asarray(m.get("val_loss", []), dtype=float))
        entry["train_loss"].append(
            np.asarray(losses_all.get(hd, {}).get("train", []), dtype=float)
        )
    # collect best metrics if present
    bt = data_key.get("metrics", {}).get("best_test_acc")
    bf = data_key.get("metrics", {}).get("best_fidelity")
    if bt is not None:
        best_test_acc_all.append(bt)
    if bf is not None:
        best_fid_all.append(bf)

hidden_dims = sorted(agg.keys())

# ------------------------------------------------------------------
# 3. Plot aggregated accuracy curves
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for hd in hidden_dims:
        train_mat = pad_to_max(agg[hd]["train_acc"])
        val_mat = pad_to_max(agg[hd]["val_acc"])
        epochs = np.arange(1, train_mat.shape[1] + 1)

        train_mean = np.nanmean(train_mat, axis=0)
        train_sem = np.nanstd(train_mat, axis=0) / np.sqrt(
            np.sum(~np.isnan(train_mat), axis=0)
        )
        val_mean = np.nanmean(val_mat, axis=0)
        val_sem = np.nanstd(val_mat, axis=0) / np.sqrt(
            np.sum(~np.isnan(val_mat), axis=0)
        )

        plt.plot(epochs, train_mean, label=f"{hd}-train mean")
        plt.fill_between(
            epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.2
        )
        plt.plot(epochs, val_mean, linestyle="--", label=f"{hd}-val mean")
        plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Mean ± SEM Training/Validation Accuracy")
    plt.legend(fontsize=7, ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_agg_acc_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4. Plot aggregated loss curves
# ------------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for hd in hidden_dims:
        train_mat = pad_to_max(agg[hd]["train_loss"])
        val_mat = pad_to_max(agg[hd]["val_loss"])
        epochs = np.arange(1, train_mat.shape[1] + 1)

        train_mean = np.nanmean(train_mat, axis=0)
        train_sem = np.nanstd(train_mat, axis=0) / np.sqrt(
            np.sum(~np.isnan(train_mat), axis=0)
        )
        val_mean = np.nanmean(val_mat, axis=0)
        val_sem = np.nanstd(val_mat, axis=0) / np.sqrt(
            np.sum(~np.isnan(val_mat), axis=0)
        )

        plt.plot(epochs, train_mean, label=f"{hd}-train mean")
        plt.fill_between(
            epochs, train_mean - train_sem, train_mean + train_sem, alpha=0.2
        )
        plt.plot(epochs, val_mean, linestyle="--", label=f"{hd}-val mean")
        plt.fill_between(epochs, val_mean - val_sem, val_mean + val_sem, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Mean ± SEM Training/Validation Loss")
    plt.legend(fontsize=7, ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_agg_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 5. Plot final validation accuracy per hidden dim (mean ± SEM)
# ------------------------------------------------------------------
try:
    means, sems = [], []
    for hd in hidden_dims:
        finals = [arr[-1] for arr in agg[hd]["val_acc"] if len(arr)]
        means.append(np.mean(finals))
        sems.append(np.std(finals) / np.sqrt(len(finals)))
    plt.figure(figsize=(5, 3))
    x = np.arange(len(hidden_dims))
    plt.bar(x, means, yerr=sems, capsize=4, color="skyblue")
    plt.xticks(x, [str(hd) for hd in hidden_dims])
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Final Val Accuracy")
    plt.title("SPR_BENCH: Final Validation Accuracy (Mean ± SEM)")
    fname = os.path.join(working_dir, "SPR_BENCH_final_val_acc_bar_agg.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated final val accuracy bar: {e}")
    plt.close()

# ------------------------------------------------------------------
# 6. Print aggregated best metrics
# ------------------------------------------------------------------
if best_test_acc_all:
    print(
        f"Aggregated Best Test Accuracy: {np.mean(best_test_acc_all):.4f} ± {np.std(best_test_acc_all)/np.sqrt(len(best_test_acc_all)):.4f}"
    )
if best_fid_all:
    print(
        f"Aggregated Rule Fidelity: {np.mean(best_fid_all):.4f} ± {np.std(best_fid_all)/np.sqrt(len(best_fid_all)):.4f}"
    )
