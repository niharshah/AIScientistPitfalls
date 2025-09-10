import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to sort cluster keys numerically (e.g. "k4" -> 4)
def sort_key(k):
    try:
        return int(k.lstrip("k"))
    except Exception:
        return 0


cluster_keys = sorted(
    [k for k in experiment_data.keys() if k.startswith("k")], key=sort_key
)

# ---------- 1. Loss curves ----------
try:
    plt.figure()
    for k in cluster_keys:
        tr = experiment_data[k]["spr_bench"]["losses"]["train"]
        vl = experiment_data[k]["spr_bench"]["losses"]["val"]
        if tr and vl:
            ep_tr, loss_tr = zip(*tr)
            ep_vl, loss_vl = zip(*vl)
            plt.plot(ep_tr, loss_tr, label=f"{k}-train")
            plt.plot(ep_vl, loss_vl, "--", label=f"{k}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Val Loss (all cluster settings)")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- 2. CSHM curves ----------
try:
    plt.figure()
    for k in cluster_keys:
        metrics = experiment_data[k]["spr_bench"]["metrics"]["val"]
        if metrics:
            ep, cwa, swa, cshm = zip(*metrics)
            plt.plot(ep, cshm, label=f"{k}")
    plt.xlabel("Epoch")
    plt.ylabel("CSHM")
    plt.title("SPR_BENCH: Color/Shape Harmonic Mean (Validation)")
    plt.legend(title="k-clusters")
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_cshm_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CSHM curve plot: {e}")
    plt.close()

# ---------- 3. Final metric bar chart ----------
try:
    final_cwa, final_swa, final_cshm = [], [], []
    labels = []
    for k in cluster_keys:
        metrics = experiment_data[k]["spr_bench"]["metrics"]["val"]
        if metrics:
            _, cwa, swa, cshm = metrics[-1]
            labels.append(k)
            final_cwa.append(cwa)
            final_swa.append(swa)
            final_cshm.append(cshm)
    x = np.arange(len(labels))
    width = 0.25
    plt.figure()
    plt.bar(x - width, final_cwa, width=width, label="CWA")
    plt.bar(x, final_swa, width=width, label="SWA")
    plt.bar(x + width, final_cshm, width=width, label="CSHM")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Final Validation Metrics by Cluster Setting")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_final_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final metrics plot: {e}")
    plt.close()

# ---------- print final metrics ----------
print("\nFinal validation metrics (last epoch per k):")
for k in cluster_keys:
    m = experiment_data[k]["spr_bench"]["metrics"]["val"]
    if m:
        ep, cwa, swa, cshm = m[-1]
        print(f"{k:>3} | CWA {cwa:.3f} | SWA {swa:.3f} | CSHM {cshm:.3f}")
