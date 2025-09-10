import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Helper: collect per-emb_dim info
runs = experiment_data.get("emb_dim", {})
emb_dims = sorted(runs.keys(), key=lambda x: int(x))
epochs_dict, val_cpx_curves = {}, {}
final_cpx, final_cwa, final_swa = [], [], []

for emb in emb_dims:
    log = runs[emb]["SPR_BENCH"]
    epochs = log["epochs"]
    val_metrics = log["metrics"]["val"]
    cpx_curve = [m["cpx"] for m in val_metrics]
    epochs_dict[emb] = epochs
    val_cpx_curves[emb] = cpx_curve
    final_cpx.append(cpx_curve[-1])
    final_cwa.append(val_metrics[-1]["cwa"])
    final_swa.append(val_metrics[-1]["swa"])

# 1) Validation CpxWA curves
try:
    plt.figure()
    for emb in emb_dims:
        plt.plot(epochs_dict[emb], val_cpx_curves[emb], marker="o", label=f"emb={emb}")
    plt.title("SPR_BENCH: Validation Complexity-WA vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_val_cpxwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA curve plot: {e}")
    plt.close()

# 2) Final CpxWA bar chart
try:
    plt.figure()
    plt.bar(emb_dims, final_cpx)
    plt.title("SPR_BENCH: Final Validation Complexity-WA by Embedding Dim")
    plt.xlabel("Embedding Dim")
    plt.ylabel("Final CpxWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_final_cpxwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final CpxWA bar plot: {e}")
    plt.close()

# 3) Final CWA bar chart
try:
    plt.figure()
    plt.bar(emb_dims, final_cwa, color="orange")
    plt.title("SPR_BENCH: Final Validation Color-WA by Embedding Dim")
    plt.xlabel("Embedding Dim")
    plt.ylabel("Final CWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_final_cwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final CWA bar plot: {e}")
    plt.close()

# 4) Final SWA bar chart
try:
    plt.figure()
    plt.bar(emb_dims, final_swa, color="green")
    plt.title("SPR_BENCH: Final Validation Shape-WA by Embedding Dim")
    plt.xlabel("Embedding Dim")
    plt.ylabel("Final SWA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_final_swa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final SWA bar plot: {e}")
    plt.close()

print("Plotting complete.")
