import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# paths / data load
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch lists of (epoch,value) as np arrays
def get_xy(run_store, key):
    # run_store['losses']['train'] etc. -> list[(epoch,val)]
    arr = np.array(run_store[key])
    return arr[:, 0], arr[:, 1]


# collect final val CoWA for summary printing / bar chart
final_cowa = {}

# ---------------------------------------------------------------
# Plot 1: Train vs Val Loss curves per embedding dimension
# ---------------------------------------------------------------
try:
    plt.figure()
    for emb_key, run in experiment_data.get("embed_dim_sweep", {}).items():
        x_tr, y_tr = get_xy(run["losses"], "train")
        x_va, y_va = get_xy(run["losses"], "val")
        plt.plot(x_tr, y_tr, marker="o", label=f"{emb_key}-train")
        plt.plot(x_va, y_va, marker="x", linestyle="--", label=f"{emb_key}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Embed Dim Sweep: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "embed_dim_sweep_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------------------------------------------------------------
# Plot 2: Train vs Val CoWA curves per embedding dimension
# ---------------------------------------------------------------
try:
    plt.figure()
    for emb_key, run in experiment_data.get("embed_dim_sweep", {}).items():
        x_tr, y_tr = get_xy(run["metrics"], "CoWA_train")
        x_va, y_va = get_xy(run["metrics"], "CoWA_val")
        plt.plot(x_tr, y_tr, marker="o", label=f"{emb_key}-train")
        plt.plot(x_va, y_va, marker="x", linestyle="--", label=f"{emb_key}-val")
        # save final val CoWA for summary
        final_cowa[emb_key] = y_va[-1] if len(y_va) else np.nan
    plt.xlabel("Epoch")
    plt.ylabel("Composite Weighted Accuracy")
    plt.title("Embed Dim Sweep: Training vs Validation CoWA")
    plt.legend()
    fname = os.path.join(working_dir, "embed_dim_sweep_cowa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CoWA curves: {e}")
    plt.close()

# ---------------------------------------------------------------
# Plot 3: Bar chart of final validation CoWA per embedding dim
# ---------------------------------------------------------------
try:
    if final_cowa:
        plt.figure()
        dims = list(final_cowa.keys())
        values = [final_cowa[d] for d in dims]
        plt.bar(dims, values, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Final Validation CoWA")
        plt.title("Embed Dim Sweep: Final Validation CoWA Summary")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "embed_dim_sweep_cowa_summary.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating CoWA summary bar chart: {e}")
    plt.close()

# ---------------------------------------------------------------
# Print final metrics
# ---------------------------------------------------------------
if final_cowa:
    print("Final Validation CoWA by embedding dimension:")
    for k, v in final_cowa.items():
        print(f"  {k}: {v:.4f}")
