import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment log -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

emb_dict = experiment_data.get("embedding_dim", {})
if not emb_dict:
    print("No embedding_dim logs found.")
    exit(0)

# ------------------ gather values ---------------------------------
dims, losses_tr, losses_val, hwa_val, swa_val, cwa_val, test_metrics = (
    [],
    {},
    {},
    {},
    {},
    {},
    {},
)

for dim_key, dim_entry in emb_dict.items():
    dim = int(dim_key.split("_")[-1])
    log = dim_entry["SPR_BENCH"]
    dims.append(dim)

    losses_tr[dim] = log["losses"]["train"]
    losses_val[dim] = log["losses"]["val"]

    h_list, s_list, c_list = [], [], []
    for m in log["metrics"]["val"]:
        h_list.append(m["HWA"])
        s_list.append(m["SWA"])
        c_list.append(m["CWA"])
    hwa_val[dim], swa_val[dim], cwa_val[dim] = h_list, s_list, c_list

    test_metrics[dim] = log["metrics"]["test"]  # dict with SWA,CWA,HWA

# sort dimensions for nicer plots
dims.sort()

# --------------------- PLOT 1: loss curves -------------------------
try:
    plt.figure()
    for dim in dims:
        epochs = list(range(1, len(losses_tr[dim]) + 1))
        plt.plot(epochs, losses_tr[dim], "--", label=f"train dim={dim}")
        plt.plot(epochs, losses_val[dim], "-", label=f"val dim={dim}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# --------------------- PLOT 2: validation HWA ----------------------
try:
    plt.figure()
    for dim in dims:
        epochs = list(range(1, len(hwa_val[dim]) + 1))
        plt.plot(epochs, hwa_val[dim], marker="o", label=f"dim={dim}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH: Validation Harmonic Weighted Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ----------- PLOT 3: final test SWA/CWA/HWA grouped bars -----------
try:
    x = np.arange(len(dims))
    width = 0.25
    swa_vals = [test_metrics[d]["SWA"] for d in dims]
    cwa_vals = [test_metrics[d]["CWA"] for d in dims]
    hwa_vals = [test_metrics[d]["HWA"] for d in dims]

    plt.figure()
    plt.bar(x - width, swa_vals, width, label="SWA")
    plt.bar(x, cwa_vals, width, label="CWA")
    plt.bar(x + width, hwa_vals, width, label="HWA")
    plt.xticks(x, [str(d) for d in dims])
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Test Metrics by Embedding Dimension")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_grouped.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating grouped bar plot: {e}")
    plt.close()

# ------------- PLOT 4: final test HWA only (highlight best) --------
try:
    plt.figure()
    plt.bar([str(d) for d in dims], hwa_vals, color="steelblue")
    best_dim = dims[int(np.argmax(hwa_vals))]
    plt.xlabel("Embedding Dimension")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH: Final Test HWA per Embedding Dim")
    fname = os.path.join(working_dir, "SPR_BENCH_test_HWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA bar plot: {e}")
    plt.close()

# ---------------------- print summary metrics ----------------------
print("Final test metrics by embedding dimension:")
for dim in dims:
    print(f"dim={dim}: {test_metrics[dim]}")
