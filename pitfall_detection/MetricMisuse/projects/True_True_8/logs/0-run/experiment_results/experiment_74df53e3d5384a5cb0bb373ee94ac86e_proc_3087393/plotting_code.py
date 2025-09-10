import matplotlib.pyplot as plt
import numpy as np
import os

# working directory for saving plots
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------- load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch nested entries
def _get(path, default=None):
    cur = experiment_data
    for k in path:
        if k in cur:
            cur = cur[k]
        else:
            return default
    return cur


spr_ft = _get(["freeze_encoder", "SPR", "fine_tune"], {})
spr_pre = _get(["freeze_encoder", "SPR", "contrastive_pretrain"], {})

# ---------------------------------------------------- 1) contrastive pre-train loss
try:
    losses = spr_pre.get("losses", [])
    if losses:
        epochs, vals = zip(*losses)
        plt.figure()
        plt.plot(epochs, vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR: Contrastive Pre-training Loss")
        save_path = os.path.join(working_dir, "SPR_contrastive_pretrain_loss.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")
except Exception as e:
    print(f"Error creating contrastive loss plot: {e}")
    plt.close()

# ---------------------------------------------------- 2) fine-tune train / val loss
try:
    tr = spr_ft.get("losses", {}).get("train", [])
    va = spr_ft.get("losses", {}).get("val", [])
    if tr and va:
        ep_tr, tr_vals = zip(*tr)
        ep_va, va_vals = zip(*va)
        plt.figure()
        plt.plot(ep_tr, tr_vals, label="Train")
        plt.plot(ep_va, va_vals, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("SPR: Fine-tune Loss (Frozen Encoder)")
        save_path = os.path.join(working_dir, "SPR_finetune_train_val_loss.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")
except Exception as e:
    print(f"Error creating fine-tune loss plot: {e}")
    plt.close()

# ---------------------------------------------------- 3) fine-tune metrics curves
try:
    swa = spr_ft.get("metrics", {}).get("SWA", [])
    cwa = spr_ft.get("metrics", {}).get("CWA", [])
    comp = spr_ft.get("metrics", {}).get("CompWA", [])
    if swa and cwa and comp:
        ep, swa_vals = zip(*swa)
        _, cwa_vals = zip(*cwa)
        _, comp_vals = zip(*comp)
        plt.figure()
        plt.plot(ep, swa_vals, label="SWA")
        plt.plot(ep, cwa_vals, label="CWA")
        plt.plot(ep, comp_vals, label="CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.title("SPR: Fine-tune Weighted-Accuracy Metrics")
        save_path = os.path.join(working_dir, "SPR_finetune_metrics.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ---------------------------------------------------- print final metrics
try:
    final_val_loss = spr_ft.get("losses", {}).get("val", [])[-1][1]
    final_swa = spr_ft.get("metrics", {}).get("SWA", [])[-1][1]
    final_cwa = spr_ft.get("metrics", {}).get("CWA", [])[-1][1]
    final_comp = spr_ft.get("metrics", {}).get("CompWA", [])[-1][1]
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(
        f"Final SWA: {final_swa:.4f} | CWA: {final_cwa:.4f} | CompWA: {final_comp:.4f}"
    )
except Exception as e:
    print(f"Could not extract final metrics: {e}")
