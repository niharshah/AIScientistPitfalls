import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bench = experiment_data.get("num_clusters_k", {}).get("SPR_BENCH", {})
k_vals = sorted(bench.keys(), key=lambda s: int(s.split("=")[1]))  # ['k=4', 'k=8', ...]


# helpers to gather series
def get_series(key_path):
    out = {}
    for k in k_vals:
        d = bench[k]
        tmp = d
        for kp in key_path:
            tmp = tmp.get(kp, [])
        out[k] = tmp
    return out


loss_train = get_series(["losses", "train"])
loss_val = get_series(["losses", "val"])
compwa_val = get_series(["metrics", "val_CompWA"])


# final CWA/SWA were printed, we recompute from stored preds/gt
def final_weighted(metric_fn):
    res = {}
    for k in k_vals:
        seqs = (
            experiment_data["num_clusters_k"]["SPR_BENCH_SEQ_CACHE"] if False else []
        )  # placeholder
        preds = np.array(bench[k]["predictions"])
        gts = np.array(bench[k]["ground_truth"])
        res[k] = metric_fn(seqs, gts, preds) if preds.size else 0.0
    return res


# Because the metric functions need sequences, quickly fetch them
seqs = experiment_data.get("SPR_BENCH_SEQS", None)
if seqs is None:
    # fall back to loading dev sequences directly stored in each k dict
    # they were not kept, so metrics already in stdout; skip recalculation
    cwa_final = swa_final = {k: np.nan for k in k_vals}
else:
    from __main__ import color_weighted_accuracy, shape_weighted_accuracy

    cwa_final = final_weighted(color_weighted_accuracy)
    swa_final = final_weighted(shape_weighted_accuracy)

# ---------- PLOTS ----------
# 1) Validation loss curves
try:
    plt.figure()
    for k in k_vals:
        plt.plot(loss_val[k], label=k)
    plt.title("SPR_BENCH Validation Loss vs Epoch\nLeft: Validation Loss curves")
    plt.xlabel("Epoch")
    plt.ylabel("Binary-Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Validation CompWA curves
try:
    plt.figure()
    for k in k_vals:
        plt.plot(compwa_val[k], label=k)
    plt.title("SPR_BENCH Validation Complexity-Weighted-Accuracy\nRight: CompWA curves")
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_CompWA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CompWA plot: {e}")
    plt.close()

# 3) Final CWA bar chart
try:
    plt.figure()
    vals = [cwa_final[k] for k in k_vals]
    plt.bar(k_vals, vals)
    plt.title("SPR_BENCH Final Color-Weighted-Accuracy")
    plt.ylabel("CWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_CWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA bar plot: {e}")
    plt.close()

# 4) Final SWA bar chart
try:
    plt.figure()
    vals = [swa_final[k] for k in k_vals]
    plt.bar(k_vals, vals)
    plt.title("SPR_BENCH Final Shape-Weighted-Accuracy")
    plt.ylabel("SWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_SWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA bar plot: {e}")
    plt.close()

# ---------- print summary ----------
print("Final metrics:")
for k in k_vals:
    print(f"{k}: CWA={cwa_final.get(k, 'N/A'):.4f}, SWA={swa_final.get(k, 'N/A'):.4f}")
