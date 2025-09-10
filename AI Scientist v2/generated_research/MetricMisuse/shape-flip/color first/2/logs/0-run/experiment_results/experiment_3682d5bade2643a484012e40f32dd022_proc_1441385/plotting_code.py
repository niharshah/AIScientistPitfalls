import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("embedding_dim", {}).get("SPR_BENCH", {})
if not spr_data:
    print("No SPR_BENCH data found.")
    exit()


# helper to collect series
def collect_series(key):
    losses_train, losses_val, dwa_val = {}, {}, {}
    for dim_key, d in spr_data.items():
        ep = np.arange(1, len(d["losses"]["train"]) + 1)
        lt = np.array([v for _, v in d["losses"]["train"]])
        lv = np.array([v for _, v in d["losses"]["val"]])
        dv = np.array([v for _, v in d["metrics"]["val"]])
        losses_train[dim_key] = (ep, lt)
        losses_val[dim_key] = (ep, lv)
        dwa_val[dim_key] = (ep, dv)
    return losses_train, losses_val, dwa_val


loss_tr, loss_val, dwa_val = collect_series(spr_data)

# -------- Figure 1: Loss curves --------
try:
    plt.figure()
    for dim_key in loss_tr:
        ep, lt = loss_tr[dim_key]
        _, lv = loss_val[dim_key]
        plt.plot(ep, lt, label=f"{dim_key}-train")
        plt.plot(ep, lv, linestyle="--", label=f"{dim_key}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Training, Right: Validation (dashed)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss figure: {e}")
    plt.close()

# -------- Figure 2: Validation Dual Weighted Accuracy --------
try:
    plt.figure()
    for dim_key in dwa_val:
        ep, dv = dwa_val[dim_key]
        plt.plot(ep, dv, marker="o", label=f"{dim_key}")
    plt.xlabel("Epoch")
    plt.ylabel("Dual Weighted Accuracy")
    plt.title("SPR_BENCH Validation DWA across Embedding Sizes")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_DWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating DWA figure: {e}")
    plt.close()

# -------- Print final DWA --------
for dim_key, (_, dv) in dwa_val.items():
    print(f"Final DWA ({dim_key}): {dv[-1]:.4f}")
