import matplotlib.pyplot as plt
import numpy as np
import os

# --------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data -------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
root = experiment_data.get("dropout_rate", {}).get(ds_key, {})


# Helper to pull a series
def tuple_to_arrays(tuples, idx_val=1):
    xs, ys = zip(*[(t[0], t[idx_val]) for t in tuples]) if tuples else ([], [])
    return np.array(xs), np.array(ys)


# 1) Loss curves ---------------------------------------------------------------
try:
    plt.figure()
    for p_key, rec in root.items():
        xs_t, ys_t = tuple_to_arrays(rec["losses"]["train"])
        xs_v, ys_v = tuple_to_arrays(rec["losses"]["val"])
        plt.plot(xs_t, ys_t, label=f"{p_key} Train")
        plt.plot(xs_v, ys_v, linestyle="--", label=f"{p_key} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) HWA over epochs -----------------------------------------------------------
try:
    plt.figure()
    for p_key, rec in root.items():
        xs, hwa = tuple_to_arrays(rec["metrics"]["val"], idx_val=3)
        plt.plot(xs, hwa, marker="o", label=p_key)
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: HWA vs Epoch for Different Dropouts")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# 3) Final-epoch HWA bar chart -------------------------------------------------
try:
    plt.figure()
    p_keys, final_hwa = [], []
    for p_key, rec in root.items():
        if rec["metrics"]["val"]:
            p_keys.append(p_key)
            final_hwa.append(rec["metrics"]["val"][-1][3])
    plt.bar(p_keys, final_hwa, color="skyblue")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_BENCH: Final HWA by Dropout Setting")
    for i, v in enumerate(final_hwa):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar chart: {e}")
    plt.close()

# ------------- summary ------------------------
if final_hwa:
    best_idx = int(np.argmax(final_hwa))
    print(f"Best dropout = {p_keys[best_idx]} with HWA={final_hwa[best_idx]:.4f}")
