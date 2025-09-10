import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------
# Load every experiment file that the system provided
# ----------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_9b13a95008444d0682d39565fc5fcd0a_proc_2991855/experiment_data.npy",
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_ae2af866218b44dc9b3b4306849fd959_proc_2991858/experiment_data.npy",
    "experiments/2025-08-15_22-25-14_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_3344104bff8f4846b93d6f37175d1293_proc_2991852/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# Helper
def _get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


# ----------------------------------------------------
# Aggregate across runs for SPR_BENCH
# ----------------------------------------------------
agg = {}  # dict[tuning_len] -> dict { 'train': [runs x T], 'val': [...], 'hwa': [...] }
for ed in all_experiment_data:
    spr_runs = _get(ed, "num_epochs_tuning", "SPR_BENCH", default={})
    if not spr_runs:
        continue
    for tuning_len, logs in spr_runs.items():
        tr = np.asarray(logs["losses"]["train"], dtype=float)
        vl = np.asarray(logs["losses"]["val"], dtype=float)
        hwa = np.asarray([m["hwa"] for m in logs["metrics"]["val"]], dtype=float)
        entry = agg.setdefault(tuning_len, {"train": [], "val": [], "hwa": []})
        entry["train"].append(tr)
        entry["val"].append(vl)
        entry["hwa"].append(hwa)

# ----------------------------------------------------
# Make sure all arrays in each group are same length
# (trim to min length so we can average safely)
# ----------------------------------------------------
for tun_len, d in agg.items():
    for k in ["train", "val", "hwa"]:
        if not d[k]:
            continue
        min_len = min(map(len, d[k]))
        d[k] = np.stack([a[:min_len] for a in d[k]], axis=0)  # shape (R, T)

# ----------------------------------------------------
# Plot 1: Train & Val losses (mean ± SEM)
# ----------------------------------------------------
try:
    if agg:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for tun_len, d in sorted(agg.items(), key=lambda x: int(x[0])):
            epochs = np.arange(1, d["train"].shape[1] + 1)
            # train
            mean_tr = d["train"].mean(axis=0)
            sem_tr = d["train"].std(axis=0, ddof=1) / np.sqrt(d["train"].shape[0])
            axes[0].plot(epochs, mean_tr, label=f"{tun_len}e – mean")
            axes[0].fill_between(epochs, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)
            # val
            mean_val = d["val"].mean(axis=0)
            sem_val = d["val"].std(axis=0, ddof=1) / np.sqrt(d["val"].shape[0])
            axes[1].plot(epochs, mean_val, label=f"{tun_len}e – mean")
            axes[1].fill_between(
                epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.2
            )
        axes[0].set_title("Left: Train Loss (mean ± SEM)")
        axes[1].set_title("Right: Validation Loss (mean ± SEM)")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.suptitle("SPR_BENCH Aggregated Loss Curves")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(working_dir, "SPR_BENCH_aggregated_loss_curves.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ----------------------------------------------------
# Plot 2: HWA (mean ± SEM)
# ----------------------------------------------------
try:
    if agg:
        plt.figure(figsize=(7, 4))
        for tun_len, d in sorted(agg.items(), key=lambda x: int(x[0])):
            epochs = np.arange(1, d["hwa"].shape[1] + 1)
            mean_hwa = d["hwa"].mean(axis=0)
            sem_hwa = d["hwa"].std(axis=0, ddof=1) / np.sqrt(d["hwa"].shape[0])
            plt.plot(epochs, mean_hwa, label=f"{tun_len}e – mean")
            plt.fill_between(epochs, mean_hwa - sem_hwa, mean_hwa + sem_hwa, alpha=0.2)
        plt.title("SPR_BENCH Validation HWA (mean ± SEM)")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_aggregated_HWA_curves.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA plot: {e}")
    plt.close()

# ----------------------------------------------------
# Plot 3: Best HWA per tuning length (mean ± SEM)
# ----------------------------------------------------
try:
    if agg:
        xs, means, sems = [], [], []
        for tun_len, d in sorted(agg.items(), key=lambda x: int(x[0])):
            best_vals = d["hwa"].max(axis=1)  # best per run
            xs.append(str(tun_len))
            means.append(best_vals.mean())
            sems.append(best_vals.std(ddof=1) / np.sqrt(len(best_vals)))
        plt.figure(figsize=(6, 4))
        plt.bar(xs, means, yerr=sems, capsize=5, alpha=0.8)
        plt.title("SPR_BENCH Best Validation HWA\n(mean ± SEM across runs)")
        plt.xlabel("Max Epochs")
        plt.ylabel("Best HWA")
        save_path = os.path.join(working_dir, "SPR_BENCH_aggregated_best_HWA.png")
        plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated best HWA bar plot: {e}")
    plt.close()

# ----------------------------------------------------
# Print out aggregated numeric arrays for quick inspection
# ----------------------------------------------------
for tun_len, d in agg.items():
    print(f"\n=== Tuning length {tun_len} epochs ===")
    print("Train loss mean (first 5):", np.round(d["train"].mean(axis=0)[:5], 4))
    print("Val   loss mean (first 5):", np.round(d["val"].mean(axis=0)[:5], 4))
    print("HWA        mean (first 5):", np.round(d["hwa"].mean(axis=0)[:5], 4))
