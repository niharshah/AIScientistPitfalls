import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment data                                               #
# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

wd_runs = experiment_data.get("weight_decay", {})
if not wd_runs:
    print("No weight-decay runs found in experiment data.")

# Collect final metrics for later summary
final_dev_metrics = []

# ------------------------------------------------------------------ #
# 1. Loss curves                                                     #
# ------------------------------------------------------------------ #
try:
    plt.figure(figsize=(8, 5))
    for wd, store in wd_runs.items():
        tr = np.array(store["losses"]["train"])
        va = np.array(store["losses"]["val"])
        plt.plot(tr[:, 0], tr[:, 1], label=f"train wd={wd}")
        plt.plot(va[:, 0], va[:, 1], linestyle="--", label=f"val wd={wd}")
    plt.title("SPR-BENCH: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    save_path = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2. Validation HCSA over epochs                                     #
# ------------------------------------------------------------------ #
try:
    plt.figure(figsize=(8, 5))
    for wd, store in wd_runs.items():
        metrics = np.array(store["metrics"]["val"])  # epoch, CWA, SWA, HCSA
        plt.plot(metrics[:, 0], metrics[:, 3], label=f"wd={wd}")
    plt.title("SPR-BENCH: Validation HCSA Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("HCSA")
    plt.legend()
    save_path = os.path.join(working_dir, "spr_bench_val_hcsa_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HCSA curves: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3. Final dev HCSA bar chart                                        #
# ------------------------------------------------------------------ #
try:
    plt.figure(figsize=(7, 4))
    wds, hcsa_vals = [], []
    for wd, store in wd_runs.items():
        last_ep, cwa, swa, hcsa = store["metrics"]["val"][-1]
        wds.append(wd)
        hcsa_vals.append(hcsa)
        final_dev_metrics.append((wd, cwa, swa, hcsa))
    plt.bar(wds, hcsa_vals, color="skyblue")
    plt.title("SPR-BENCH: Final Dev HCSA by Weight Decay")
    plt.xlabel("Weight Decay")
    plt.ylabel("HCSA")
    save_path = os.path.join(working_dir, "spr_bench_final_dev_hcsa_bar.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating final HCSA bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4. CWA vs SWA scatter (dev set)                                    #
# ------------------------------------------------------------------ #
try:
    plt.figure(figsize=(6, 6))
    for wd, cwa, swa, hcsa in final_dev_metrics:
        plt.scatter(cwa, swa, label=f"wd={wd}")
        plt.text(cwa + 0.002, swa + 0.002, wd, fontsize=8)
    plt.title("SPR-BENCH Dev Set: CWA vs SWA (best epoch)")
    plt.xlabel("Color-Weighted Accuracy (CWA)")
    plt.ylabel("Shape-Weighted Accuracy (SWA)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.3)
    save_path = os.path.join(working_dir, "spr_bench_cwa_vs_swa_scatter.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating CWA vs SWA scatter: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 5. Print best weight-decay according to dev HCSA                   #
# ------------------------------------------------------------------ #
if final_dev_metrics:
    best_wd, _, _, best_hcsa = max(final_dev_metrics, key=lambda x: x[3])
    print(f"Best weight-decay on dev set: {best_wd} with HCSA={best_hcsa:.3f}")
