import matplotlib.pyplot as plt
import numpy as np
import os

# --------- paths ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data -----
experiment_data_path_list = [
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_8ee2756a7cf54ee592c35173982c8f4e_proc_3085142/experiment_data.npy",
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_e9de10b9c92741489c16dc6cf9ee5f9f_proc_3085141/experiment_data.npy",
    "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_ef1c88b6c83341b6b85b18ea7a9b63c1_proc_3085140/experiment_data.npy",
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

dataset_name = "SPR_dataset"  # generic tag


def unpack(store, key_path):
    """Return epochs, vals (np arrays) for a single run; empty arrays if missing."""
    cur = store
    for k in key_path:
        cur = cur.get(k, [])
    if not cur:
        return np.array([]), np.array([])
    ep, val = zip(*cur)
    return np.array(ep), np.array(val)


def aggregate_runs(all_runs, key_path):
    """Collect values from every run, align by epoch, return epoch, mean, sem."""
    epoch_to_vals = {}
    for run in all_runs:
        ep, vals = unpack(run, key_path)
        for e, v in zip(ep, vals):
            epoch_to_vals.setdefault(e, []).append(v)
    if not epoch_to_vals:
        return np.array([]), np.array([]), np.array([])
    epochs = np.array(sorted(epoch_to_vals))
    means = np.array([np.mean(epoch_to_vals[e]) for e in epochs])
    sems = np.array(
        [
            np.std(epoch_to_vals[e], ddof=1) / np.sqrt(len(epoch_to_vals[e]))
            for e in epochs
        ]
    )
    return epochs, means, sems


plot_id = 0
max_plots = 5

# 1) contrastive loss (mean ± SEM)
if plot_id < max_plots:
    try:
        ep, mean_loss, sem_loss = aggregate_runs(
            all_experiment_data, ("contrastive_pretrain", "losses")
        )
        if ep.size:
            plt.figure()
            plt.errorbar(ep, mean_loss, yerr=sem_loss, fmt="-o", label="Mean ± SEM")
            plt.fill_between(ep, mean_loss - sem_loss, mean_loss + sem_loss, alpha=0.2)
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            plt.title(f"Contrastive Pretrain Loss ({dataset_name})")
            plt.legend()
            fname = f"{dataset_name}_contrastive_loss_mean_sem.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plot_id += 1
    except Exception as e:
        print(f"Error plotting aggregated contrastive loss: {e}")
        plt.close()

# 2) fine-tune loss curves (train & val) mean ± SEM
if plot_id < max_plots:
    try:
        ep_tr, mean_tr, sem_tr = aggregate_runs(
            all_experiment_data, ("fine_tune", "losses", "train")
        )
        ep_va, mean_va, sem_va = aggregate_runs(
            all_experiment_data, ("fine_tune", "losses", "val")
        )
        if ep_tr.size and ep_va.size:
            plt.figure()
            plt.errorbar(
                ep_tr, mean_tr, yerr=sem_tr, fmt="-o", label="Train Mean ± SEM"
            )
            plt.fill_between(ep_tr, mean_tr - sem_tr, mean_tr + sem_tr, alpha=0.2)
            plt.errorbar(ep_va, mean_va, yerr=sem_va, fmt="-s", label="Val Mean ± SEM")
            plt.fill_between(ep_va, mean_va - sem_va, mean_va + sem_va, alpha=0.2)
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"Fine-tune Loss Curves ({dataset_name})")
            plt.legend()
            fname = f"{dataset_name}_finetune_loss_mean_sem.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plot_id += 1
    except Exception as e:
        print(f"Error plotting aggregated fine-tune loss: {e}")
        plt.close()

# 3-5) weighted accuracy metrics
metric_names = {
    "SWA": "Shape-Weighted Acc",
    "CWA": "Color-Weighted Acc",
    "CompWA": "Complexity-Weighted Acc",
}

for m_key, m_title in metric_names.items():
    if plot_id >= max_plots:
        break
    try:
        ep, mean_vals, sem_vals = aggregate_runs(
            all_experiment_data, ("fine_tune", "metrics", m_key)
        )
        if ep.size:
            plt.figure()
            plt.errorbar(ep, mean_vals, yerr=sem_vals, fmt="-o", label="Mean ± SEM")
            plt.fill_between(ep, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.2)
            plt.xlabel("Epoch")
            plt.ylabel(m_title)
            plt.title(
                f"{m_title} over Epochs ({dataset_name})\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.legend()
            fname = f"{dataset_name}_{m_key}_mean_sem_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plot_id += 1
    except Exception as e:
        print(f"Error plotting aggregated {m_key}: {e}")
        plt.close()

# --------- print final aggregated metrics -----------
try:
    final_val_losses = []
    final_SWA, final_CWA, final_CompWA = [], [], []
    for run in all_experiment_data:
        val_loss = unpack(run, ("fine_tune", "losses", "val"))[1]
        if val_loss.size:
            final_val_losses.append(val_loss[-1])
        swa = unpack(run, ("fine_tune", "metrics", "SWA"))[1]
        cwa = unpack(run, ("fine_tune", "metrics", "CWA"))[1]
        compwa = unpack(run, ("fine_tune", "metrics", "CompWA"))[1]
        if swa.size:
            final_SWA.append(swa[-1])
        if cwa.size:
            final_CWA.append(cwa[-1])
        if compwa.size:
            final_CompWA.append(compwa[-1])

    if final_val_losses:
        print(
            f"Final Val Loss: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses, ddof=1)/np.sqrt(len(final_val_losses)):.4f} | "
            f"SWA: {np.mean(final_SWA):.4f} ± {np.std(final_SWA, ddof=1)/np.sqrt(len(final_SWA)):.4f} | "
            f"CWA: {np.mean(final_CWA):.4f} ± {np.std(final_CWA, ddof=1)/np.sqrt(len(final_CWA)):.4f} | "
            f"CompWA: {np.mean(final_CompWA):.4f} ± {np.std(final_CompWA, ddof=1)/np.sqrt(len(final_CompWA)):.4f}"
        )
except Exception as e:
    print(f"Error printing aggregated final metrics: {e}")
