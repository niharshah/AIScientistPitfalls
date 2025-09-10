import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- set up paths -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_49dc8a268947475f9f564fd114e83e6f_proc_3098926/experiment_data.npy",
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_a12ea8dec414461ab38c0a798bb2a4f3_proc_3098928/experiment_data.npy",
    "experiments/2025-08-16_02-32-02_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_fbf6902180b548959741515e842a3c30_proc_3098927/experiment_data.npy",
]

# --------------------- load data from all runs -------------------------------------
all_bench = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        d = np.load(full_path, allow_pickle=True).item()
        all_bench.append(d["embedding_dim"]["SPR_BENCH"])
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_bench = []

if len(all_bench) == 0:
    print("No experiment data available – aborting plotting.")
    exit()


# --------------------- helper : stack & trim to same length -------------------------
def stack_and_trim(key1, key2):
    """
    key1 = 'losses' | 'metrics'
    key2 = 'train' , 'val' , 'SWA', ...
    returns stacked np.array (runs, epochs)
    """
    arrays = [np.asarray(b[key1][key2]) for b in all_bench]
    min_len = min([a.shape[0] for a in arrays])
    arrays = [a[:min_len] for a in arrays]
    return np.stack(arrays, axis=0)


# common epoch axis
train_loss_mat = stack_and_trim("losses", "train")
val_loss_mat = stack_and_trim("losses", "val")
epochs = np.arange(1, train_loss_mat.shape[1] + 1)

# --------------------- Plot 1 : Loss curves (mean ± SE) -----------------------------
try:
    plt.figure()
    for arr, label, c in [
        (train_loss_mat, "Train Loss", "tab:blue"),
        (val_loss_mat, "Validation Loss", "tab:orange"),
    ]:
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        plt.plot(epochs, mean, label=f"{label} (mean)")
        plt.fill_between(
            epochs, mean - se, mean + se, alpha=0.2, color=c, label=f"{label} (±SE)"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH (embedding_dim): Train vs Validation Loss – Mean ± SE")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves_mean_se.png")
    plt.savefig(fname)
    print("Saved", fname)
    # print last epoch val loss aggregate
    print(
        f"Final validation loss: {val_loss_mat[:,-1].mean():.4f} ± "
        f"{val_loss_mat[:,-1].std(ddof=1)/np.sqrt(val_loss_mat.shape[0]):.4f}"
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# --------------------- Plot 2 : Metric curves (mean ± SE) ---------------------------
try:
    metric_names = ["SWA", "CWA", "HWA"]
    colors = ["tab:green", "tab:red", "tab:purple"]
    plt.figure()
    for m, col in zip(metric_names, colors):
        mat = stack_and_trim("metrics", m)
        mean = mat.mean(axis=0)
        se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        plt.plot(epochs, mean, color=col, label=f"{m} (mean)")
        plt.fill_between(
            epochs, mean - se, mean + se, color=col, alpha=0.2, label=f"{m} (±SE)"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH (embedding_dim): Weighted Accuracies – Mean ± SE")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_metric_curves_mean_se.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metric plot: {e}")
    plt.close()

# --------------------- Plot 3 : Accuracy vs Embedding Dim (mean ± SE) ---------------
try:
    embed_dims = all_bench[0]["config_values"]  # assume identical across runs
    per_run_acc = []
    for bench in all_bench:
        acc_this_run = []
        for gt, pr in zip(bench["ground_truth"], bench["predictions"]):
            gt = np.asarray(gt)
            pr = np.asarray(pr)
            acc_this_run.append((gt == pr).mean())
        per_run_acc.append(acc_this_run)
    acc_mat = np.stack(per_run_acc, axis=0)  # shape (runs, dims)
    means = acc_mat.mean(axis=0)
    se = acc_mat.std(axis=0, ddof=1) / np.sqrt(acc_mat.shape[0])

    x = np.arange(len(embed_dims))
    plt.figure()
    plt.bar(x, means, yerr=se, capsize=5, tick_label=embed_dims, color="tab:cyan")
    plt.ylim(0, 1)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Final Epoch Accuracy")
    plt.title("SPR_BENCH: Accuracy vs Embedding Size – Mean ± SE")
    fname = os.path.join(working_dir, "spr_bench_accuracy_by_dim_mean_se.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()
