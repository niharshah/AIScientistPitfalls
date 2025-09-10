import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- list of experiment files ----------
experiment_data_path_list = [
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_f8d17a8453644807b6265ed7a71263a1_proc_2989518/experiment_data.npy",
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_6f8aa4ef9909408695ee9fcd5dde0c78_proc_2989519/experiment_data.npy",
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_df5bbf6cd848448fa6cc106d80343911_proc_2989517/experiment_data.npy",
]

# ---------- load all experiments ----------
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

dataset_name = "SPR_BENCH"


# ---------- helper functions ----------
def _style(idx):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)], "-" if idx < len(colors) else "--"


def _collect_across_runs(key_level_3, subsection):
    """
    key_level_3 : e.g. 'losses' or 'metrics'
    subsection  : e.g. 'pretrain', 'train', 'val', 'SWA'...
    returns dict: {PT_k : [run1_arr, run2_arr, ...]}
    """
    out = {}
    for exp in all_experiment_data:
        try:
            pt_dict = exp.get("pretrain_epochs", {}).get(dataset_name, {})
            for k, info in pt_dict.items():
                arr = info.get(key_level_3, {}).get(subsection, [])
                if arr:
                    out.setdefault(k, []).append(np.asarray(arr))
        except Exception:
            continue
    return out


def _mean_sem(list_of_arrays):
    # cut to shortest run length
    min_len = min(len(a) for a in list_of_arrays)
    arr = np.stack([a[:min_len] for a in list_of_arrays], axis=0)  # (n_runs, T)
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, sem


# ---------- aggregated plots ----------
# 1. Pre-training loss
try:
    plt.figure()
    data_dict = _collect_across_runs("losses", "pretrain")
    for i, k in enumerate(sorted(data_dict.keys(), key=lambda x: int(x))):
        mean, sem = _mean_sem(data_dict[k])
        xs = np.arange(1, len(mean) + 1)
        c, ls = _style(i)
        plt.plot(xs, mean, color=c, linestyle=ls, label=f"PT={k} mean")
        plt.fill_between(xs, mean - sem, mean + sem, color=c, alpha=0.25)
    plt.title("SPR_BENCH: Pre-training Loss (mean ± SEM)")
    plt.xlabel("Pre-training Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss_aggregated.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated pre-training loss plot: {e}")
    plt.close()

# 2. Fine-tuning loss (train+val)
try:
    plt.figure()
    train_dict = _collect_across_runs("losses", "train")
    val_dict = _collect_across_runs("losses", "val")
    for i, k in enumerate(sorted(train_dict.keys(), key=lambda x: int(x))):
        if k not in val_dict:
            continue
        # train
        mean_t, sem_t = _mean_sem(train_dict[k])
        xs = np.arange(1, len(mean_t) + 1)
        c, _ = _style(i)
        plt.plot(xs, mean_t, color=c, linestyle="-", label=f"Train PT={k}")
        plt.fill_between(xs, mean_t - sem_t, mean_t + sem_t, color=c, alpha=0.20)
        # val
        mean_v, sem_v = _mean_sem(val_dict[k])
        plt.plot(xs, mean_v, color=c, linestyle="--", label=f"Val PT={k}")
        plt.fill_between(xs, mean_v - sem_v, mean_v + sem_v, color=c, alpha=0.10)
    plt.title("SPR_BENCH: Fine-tuning Loss (mean ± SEM)")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize="small", ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_finetune_loss_aggregated.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated fine-tuning loss plot: {e}")
    plt.close()

# 3-5. Metrics
for metric in ["SWA", "CWA", "SCHM"]:
    try:
        plt.figure()
        m_dict = _collect_across_runs("metrics", metric)
        for i, k in enumerate(sorted(m_dict.keys(), key=lambda x: int(x))):
            mean, sem = _mean_sem(m_dict[k])
            xs = np.arange(1, len(mean) + 1)
            c, ls = _style(i)
            plt.plot(xs, mean, color=c, linestyle=ls, label=f"PT={k} mean")
            plt.fill_between(xs, mean - sem, mean + sem, color=c, alpha=0.25)
            # record final value for console
            if xs[-1] == len(mean):
                print(f"{metric} final mean PT={k}: {mean[-1]:.4f} ± {sem[-1]:.4f}")
        plt.title(f"SPR_BENCH: {metric} (mean ± SEM)")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel(metric)
        plt.legend(fontsize="small")
        fname = os.path.join(working_dir, f"SPR_BENCH_{metric}_aggregated.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated {metric} plot: {e}")
        plt.close()
