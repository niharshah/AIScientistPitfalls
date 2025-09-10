import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup & load --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# detect k values (expect keys like 'k=8')
k_vals = sorted(
    [k for k in experiment_data if k.startswith("k=")],
    key=lambda s: int(s.split("=")[1]),
)
if not k_vals:
    print("No k=... experiments found. Exiting.")
    exit()


# -------- helper to stack series --------
def collect(path):
    out = {}
    for k in k_vals:
        d = experiment_data[k]
        for p in path:
            d = d.get(p, [])
        out[k] = np.asarray(d)
    return out


loss_train = collect(["losses", "train"])
loss_val = collect(["losses", "val"])
cwa = collect(["metrics", "val", "CWA"])
swa = collect(["metrics", "val", "SWA"])
comp = collect(["metrics", "val", "CompWA"])

# final metrics (last epoch)
final_cwa = {k: arr[-1] if arr.size else np.nan for k, arr in cwa.items()}
final_swa = {k: arr[-1] if arr.size else np.nan for k, arr in swa.items()}
final_comp = {k: arr[-1] if arr.size else np.nan for k, arr in comp.items()}

# -------- PLOTS --------
# 1) train/val loss curves
try:
    plt.figure()
    for k in k_vals:
        plt.plot(loss_train[k], "--", label=f"{k} train")
        plt.plot(loss_val[k], "-", label=f"{k} val")
    plt.title("SPR_BENCH Loss vs Epoch\nLeft: train (dashed), Right: val (solid)")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) CompWA curves
try:
    plt.figure()
    for k in k_vals:
        plt.plot(comp[k], label=k)
    plt.title("SPR_BENCH Validation Complexity-Weighted-Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_CompWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CompWA curves: {e}")
    plt.close()

# 3) final metric bar chart
try:
    x = np.arange(len(k_vals))
    width = 0.25
    plt.figure()
    plt.bar(x - width, [final_cwa[k] for k in k_vals], width, label="CWA")
    plt.bar(x, [final_swa[k] for k in k_vals], width, label="SWA")
    plt.bar(x + width, [final_comp[k] for k in k_vals], width, label="CompWA")
    plt.xticks(x, k_vals)
    plt.title("SPR_BENCH Final Weighted Accuracies")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_final_weighted_accuracies.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final metrics bar chart: {e}")
    plt.close()

# -------- print summary --------
print("Final metrics per k:")
for k in k_vals:
    print(
        f"{k:>4}: CWA={final_cwa[k]:.4f}, SWA={final_swa[k]:.4f}, CompWA={final_comp[k]:.4f}"
    )
