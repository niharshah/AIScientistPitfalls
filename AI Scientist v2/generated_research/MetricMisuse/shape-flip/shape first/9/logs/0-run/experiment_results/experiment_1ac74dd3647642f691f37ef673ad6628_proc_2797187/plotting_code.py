import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    dims_dict = experiment_data["embedding_dim"]["SPR_BENCH"]
except Exception as e:
    print(f"Unable to load experiment data: {e}")
    dims_dict = {}

# ---------- plot 1: validation HWA curves ----------
try:
    if dims_dict:
        plt.figure()
        for dim, rec in dims_dict.items():
            hwa_curve = rec["metrics"]["val"]
            plt.plot(range(1, len(hwa_curve) + 1), hwa_curve, label=f"dim={dim}")
        plt.title("SPR_BENCH – Validation HWA vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_over_epochs.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ---------- identify best dim ----------
best_dim, best_final_hwa = None, -1
for dim, rec in dims_dict.items():
    if rec["metrics"]["val"]:
        if rec["metrics"]["val"][-1] > best_final_hwa:
            best_final_hwa = rec["metrics"]["val"][-1]
            best_dim = dim

# ---------- plot 2: loss curves for best dim ----------
try:
    if best_dim is not None:
        rec = dims_dict[best_dim]
        epochs = range(1, len(rec["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, rec["losses"]["train"], label="Train Loss")
        plt.plot(epochs, rec["losses"]["val"], label="Validation Loss")
        plt.title(f"SPR_BENCH – Loss Curves (Best Dim={best_dim})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"SPR_BENCH_loss_curves_dim_{best_dim}.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- plot 3: bar chart of best HWA per dim ----------
try:
    if dims_dict:
        dims = []
        best_hwas = []
        for dim, rec in dims_dict.items():
            if rec["metrics"]["val"]:
                dims.append(int(dim))
                best_hwas.append(max(rec["metrics"]["val"]))
        plt.figure()
        plt.bar([str(d) for d in dims], best_hwas, color="skyblue")
        plt.title("SPR_BENCH – Best Validation HWA per Embedding Dim")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Best HWA")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_best_HWA_per_dim.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()
