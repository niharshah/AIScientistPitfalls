import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def safe_load(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        return None


exp_path = os.path.join(working_dir, "experiment_data.npy")
exp = safe_load(exp_path)
if exp is None:
    exit()

run = exp.get("no_projector_head", {}).get("spr", {})

# 1) Contrastive pre-train loss
try:
    pts = run.get("contrastive_pretrain", {}).get("losses", [])
    if pts:
        ep, loss = zip(*pts)
        plt.figure()
        plt.plot(ep, loss, marker="o")
        plt.title("SPR Dataset – Contrastive Pre-training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        fname = os.path.join(working_dir, "spr_contrastive_pretrain_loss.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating contrastive plot: {e}")
    plt.close()

# 2) Fine-tuning train / val loss
try:
    tr = run.get("fine_tune", {}).get("losses", {}).get("train", [])
    va = run.get("fine_tune", {}).get("losses", {}).get("val", [])
    if tr and va:
        ep_tr, tr_loss = zip(*tr)
        ep_va, va_loss = zip(*va)
        plt.figure()
        plt.plot(ep_tr, tr_loss, label="Train")
        plt.plot(ep_va, va_loss, label="Validation")
        plt.title("SPR Dataset – Fine-tuning Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_finetune_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 3) Metrics curves
try:
    m = run.get("fine_tune", {}).get("metrics", {})
    swa = m.get("SWA", [])
    cwa = m.get("CWA", [])
    comp = m.get("CompWA", [])
    if swa and cwa and comp:
        ep, swa_v = zip(*swa)
        _, cwa_v = zip(*cwa)
        _, comp_v = zip(*comp)
        plt.figure()
        plt.plot(ep, swa_v, label="SWA")
        plt.plot(ep, cwa_v, label="CWA")
        plt.plot(ep, comp_v, label="CompWA")
        plt.title("SPR Dataset – Weighted Accuracy Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_weighted_accuracy_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# Quick textual summary
try:
    last_ep = max(ep) if "ep" in locals() else None
    if last_ep:
        print(
            f"Final epoch ({last_ep}) metrics – "
            f"SWA:{swa_v[-1]:.3f}  CWA:{cwa_v[-1]:.3f}  CompWA:{comp_v[-1]:.3f}"
        )
except Exception:
    pass
