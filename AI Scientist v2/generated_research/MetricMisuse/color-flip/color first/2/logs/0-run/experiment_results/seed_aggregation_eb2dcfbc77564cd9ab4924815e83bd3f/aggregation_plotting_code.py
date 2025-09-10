import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- working directory ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- collect all experiment paths ----------
experiment_data_path_list = [
    "experiments/2025-08-31_02-26-55_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_b6b0364c3fe14110aa7e8dfa579402b0_proc_1604461/experiment_data.npy",
    "experiments/2025-08-31_02-26-55_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_8081a9d9145a4ce49de09ebbda7b3d36_proc_1604462/experiment_data.npy",
    "experiments/2025-08-31_02-26-55_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_2b39d2fb89c748c594c81245f2e9b40d_proc_1604459/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        data = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------- aggregate data ----------
train_runs, val_runs, metric_runs = {}, {}, {}  # keyed by emb_dim
metric_names = set()
for data in all_experiment_data:
    exp = data.get("embedding_dim_tuning", {}).get("SPR_BENCH", {})
    for k, v in exp.items():
        emb_dim = int(k.split("_")[-1])
        # losses
        train_runs.setdefault(emb_dim, []).append(np.array(v["losses"]["train"]))
        val_runs.setdefault(emb_dim, []).append(np.array(v["losses"]["val"]))
        # metrics
        metric_runs.setdefault(emb_dim, {})
        for m_name, m_val in v["metrics"]["test"].items():
            metric_names.add(m_name)
            metric_runs[emb_dim].setdefault(m_name, []).append(m_val)

emb_dims = sorted(train_runs.keys())
metric_names = sorted(metric_names)


def mean_sem(arr_list):
    """Stack (after trimming to shortest length) and return mean and sem."""
    min_len = min(len(a) for a in arr_list)
    stack = np.stack([a[:min_len] for a in arr_list], axis=0)
    mean = stack.mean(axis=0)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    return mean, sem


# ---------- plot training loss ----------
try:
    plt.figure()
    for ed in emb_dims:
        mean, sem = mean_sem(train_runs[ed])
        epochs = np.arange(1, len(mean) + 1)
        plt.plot(epochs, mean, label=f"emb_dim={ed}")
        plt.fill_between(epochs, mean - sem, mean + sem, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("SPR_BENCH Training Loss (Mean ± SEM)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_training_loss_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training loss plot: {e}")
    plt.close()

# ---------- plot validation loss ----------
try:
    plt.figure()
    for ed in emb_dims:
        mean, sem = mean_sem(val_runs[ed])
        epochs = np.arange(1, len(mean) + 1)
        plt.plot(epochs, mean, label=f"emb_dim={ed}")
        plt.fill_between(epochs, mean - sem, mean + sem, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("SPR_BENCH Validation Loss (Mean ± SEM)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_loss_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated validation loss plot: {e}")
    plt.close()

# ---------- plot test metrics ----------
try:
    x = np.arange(len(emb_dims))
    width = 0.25
    plt.figure()
    for i, m in enumerate(metric_names):
        means, sems = [], []
        for ed in emb_dims:
            vals = metric_runs.get(ed, {}).get(m, [])
            means.append(np.mean(vals) if vals else np.nan)
            sems.append(
                np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            )
        plt.bar(
            x + (i - (len(metric_names) - 1) / 2) * width,
            means,
            width,
            yerr=sems,
            capsize=3,
            label=m,
        )
    plt.xticks(x, [str(ed) for ed in emb_dims])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(
        "SPR_BENCH Test Metrics (Mean ± SEM)\nLeft: CWA, Center: SWA, Right: GCWA"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics_mean_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test metrics plot: {e}")
    plt.close()

# ---------- print aggregated results ----------
for ed in emb_dims:
    print(f"\nEmbedding Dimension {ed}:")
    for m in metric_names:
        vals = metric_runs.get(ed, {}).get(m, [])
        if vals:
            mean_val = np.mean(vals)
            sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            print(f"  {m}: {mean_val:.4f} ± {sem_val:.4f} (n={len(vals)})")
        else:
            print(f"  {m}: No data")
