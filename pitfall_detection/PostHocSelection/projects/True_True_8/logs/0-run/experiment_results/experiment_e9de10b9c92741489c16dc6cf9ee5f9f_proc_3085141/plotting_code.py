import matplotlib.pyplot as plt
import numpy as np
import os

# --------- paths ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "SPR_dataset"  # generic tag


def unpack(store, key_path):
    """key_path=('contrastive_pretrain','losses') -> epochs, vals"""
    cur = store
    for k in key_path:
        cur = cur.get(k, [])
    if not cur:
        return np.array([]), np.array([])
    ep, val = zip(*cur)
    return np.array(ep), np.array(val)


plot_id = 0
max_plots = 5

# 1) contrastive loss
if plot_id < max_plots:
    try:
        ep, loss = unpack(experiment_data, ("contrastive_pretrain", "losses"))
        if ep.size:
            plt.figure()
            plt.plot(ep, loss, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            plt.title(f"Contrastive Pretrain Loss ({dataset_name})")
            fname = f"{dataset_name}_contrastive_loss.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plot_id += 1
    except Exception as e:
        print(f"Error plotting contrastive loss: {e}")
        plt.close()

# 2) fine-tune losses
if plot_id < max_plots:
    try:
        ep_tr, tr = unpack(experiment_data, ("fine_tune", "losses", "train"))
        ep_va, va = unpack(experiment_data, ("fine_tune", "losses", "val"))
        if ep_tr.size and ep_va.size:
            plt.figure()
            plt.plot(ep_tr, tr, label="Train")
            plt.plot(ep_va, va, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"Fine-tune Loss Curves ({dataset_name})")
            plt.legend()
            fname = f"{dataset_name}_finetune_loss.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plot_id += 1
    except Exception as e:
        print(f"Error plotting fine-tune loss: {e}")
        plt.close()

# helper for metric plots
metric_names = {
    "SWA": "Shape-Weighted Acc",
    "CWA": "Color-Weighted Acc",
    "CompWA": "Complexity-Weighted Acc",
}

for m_key, m_title in metric_names.items():
    if plot_id >= max_plots:
        break
    try:
        ep, vals = unpack(experiment_data, ("fine_tune", "metrics", m_key))
        if ep.size:
            plt.figure()
            plt.plot(ep, vals, marker="s")
            plt.xlabel("Epoch")
            plt.ylabel(m_title)
            plt.title(
                f"{m_title} over Epochs ({dataset_name})\nLeft: Ground Truth, Right: Generated Samples"
            )
            fname = f"{dataset_name}_{m_key}_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plot_id += 1
    except Exception as e:
        print(f"Error plotting {m_key}: {e}")
        plt.close()

# --------- print final metrics -----------
try:
    final_val_loss = unpack(experiment_data, ("fine_tune", "losses", "val"))[1][-1]
    final_SWA = unpack(experiment_data, ("fine_tune", "metrics", "SWA"))[1][-1]
    final_CWA = unpack(experiment_data, ("fine_tune", "metrics", "CWA"))[1][-1]
    final_CompWA = unpack(experiment_data, ("fine_tune", "metrics", "CompWA"))[1][-1]
    print(
        f"Final Val Loss: {final_val_loss:.4f}  SWA: {final_SWA:.4f}  "
        f"CWA: {final_CWA:.4f}  CompWA: {final_CompWA:.4f}"
    )
except Exception as e:
    print(f"Error printing final metrics: {e}")
