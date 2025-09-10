import matplotlib.pyplot as plt
import numpy as np
import os

# setup paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = {}

# --------- Plot 1: Pre-training contrastive loss ----------
try:
    losses_pre = spr.get("pretrain", {}).get("loss", [])
    if losses_pre:
        plt.figure(figsize=(6, 4))
        x = np.arange(1, len(losses_pre) + 1)
        plt.plot(x, losses_pre, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        plt.title("SPR_BENCH Pre-training Loss (Contrastive)")
        fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("No pre-training loss data found.")
except Exception as e:
    print(f"Error creating pre-training plot: {e}")
finally:
    plt.close()

# --------- Plot 2: Fine-tuning train vs val loss ----------
try:
    ft = spr.get("finetune", {})
    tr, vl = ft.get("loss_train", []), ft.get("loss_val", [])
    if tr and vl:
        plt.figure(figsize=(6, 4))
        x = np.arange(1, len(tr) + 1)
        plt.plot(x, tr, "o--", label="Train")
        plt.plot(x, vl, "o-", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Fine-tuning Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_finetune_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("No fine-tuning loss data to plot.")
except Exception as e:
    print(f"Error creating fine-tuning loss plot: {e}")
finally:
    plt.close()

# --------- Plot 3: Weighted accuracies ----------
try:
    swa, cwa, coa = (
        ft.get("SWA", []),
        ft.get("CWA", []),
        ft.get("CoWA", []),
    )
    if swa and cwa and coa:
        plt.figure(figsize=(6, 4))
        x = np.arange(1, len(swa) + 1)
        plt.plot(x, swa, "o-", label="SWA")
        plt.plot(x, cwa, "s-", label="CWA")
        plt.plot(x, coa, "d-", label="CoWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Weighted Accuracies During Fine-tuning")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_weighted_accuracies.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("No accuracy data to plot.")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
finally:
    plt.close()
