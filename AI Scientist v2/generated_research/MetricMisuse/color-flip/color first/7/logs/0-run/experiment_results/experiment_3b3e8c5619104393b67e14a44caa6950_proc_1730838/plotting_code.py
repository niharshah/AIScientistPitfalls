import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    raise RuntimeError(f"Could not load experiment data: {e}")

spr_data = experiment_data.get("fc_hidden_dim", {}).get("SPR_BENCH", {})
if not spr_data:
    raise ValueError("No SPR_BENCH data found in experiment_data.npy")

# collect per-hidden_dim arrays ------------------------------------------------
hid_dims, losses_tr, cpx_tr, cpx_val, epochs = [], {}, {}, {}, {}
for h, store in spr_data.items():
    hid = int(h)
    hid_dims.append(hid)
    ep = np.array(store["epochs"])
    losses_tr[hid] = np.array(store["losses"]["train"])
    cpx_tr[hid] = np.array([m["cpx"] for m in store["metrics"]["train"]])
    cpx_val[hid] = np.array([m["cpx"] for m in store["metrics"]["val"]])
    epochs[hid] = ep
hid_dims = sorted(hid_dims)

# 1) Training loss curves ------------------------------------------------------
try:
    plt.figure()
    for hid in hid_dims:
        plt.plot(epochs[hid], losses_tr[hid], marker="o", label=f"hid={hid}")
    plt.title("SPR_BENCH – Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fn = os.path.join(working_dir, "spr_bench_train_loss_curves.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error plotting training loss curves: {e}")
    plt.close()

# 2) Validation CpxWA curves ---------------------------------------------------
try:
    plt.figure()
    for hid in hid_dims:
        plt.plot(epochs[hid], cpx_val[hid], marker="o", label=f"hid={hid}")
    plt.title("SPR_BENCH – Validation Complexity-Weighted Acc.")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fn = os.path.join(working_dir, "spr_bench_val_cpxwa_curves.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error plotting val CpxWA curves: {e}")
    plt.close()

# identify best hidden_dim -----------------------------------------------------
best_hid = max(hid_dims, key=lambda h: cpx_val[h][-1])

# 3) Train vs Val CpxWA for best hidden_dim ------------------------------------
try:
    plt.figure()
    plt.plot(epochs[best_hid], cpx_tr[best_hid], marker="o", label="Train")
    plt.plot(epochs[best_hid], cpx_val[best_hid], marker="s", label="Val")
    plt.title(f"SPR_BENCH – CpxWA (hid={best_hid})")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fn = os.path.join(working_dir, f"spr_bench_cpxwa_hid{best_hid}.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error plotting best hid CpxWA: {e}")
    plt.close()

# 4) Bar chart of best Val CpxWA per hidden_dim --------------------------------
try:
    plt.figure()
    best_vals = [cpx_val[h][-1] for h in hid_dims]
    plt.bar([str(h) for h in hid_dims], best_vals)
    plt.title("SPR_BENCH – Best Validation CpxWA by Hidden Dim")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Best Val CpxWA")
    fn = os.path.join(working_dir, "spr_bench_best_val_cpxwa_bar.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error plotting best-val bar chart: {e}")
    plt.close()

print("All plots saved to", working_dir)
