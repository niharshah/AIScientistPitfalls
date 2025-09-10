import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# guard for missing data
if not experiment_data:
    exit()

exp = experiment_data["batch_size_tuning"]["SPR_BENCH"]
b_sizes = exp["batch_sizes"]
epochs_ls = exp["epochs"]  # list of epoch index lists
train_loss = exp["losses"]["train"]  # list[list]
val_mets = exp["metrics"]["val"]  # list[list[dict]]


# helper: extract per-batch-size arrays
def per_bs_metric(metric_key):
    out = []
    for bs_idx in range(len(b_sizes)):
        out.append([met[metric_key] for met in val_mets[bs_idx]])
    return out


val_cpx = per_bs_metric("cpx")
val_cwa = per_bs_metric("cwa")
val_swa = per_bs_metric("swa")

# ------------ Plot 1: Val CpxWA vs epoch ----------
try:
    plt.figure()
    for idx, bs in enumerate(b_sizes):
        plt.plot(epochs_ls[idx], val_cpx[idx], marker="o", label=f"bs={bs}")
    plt.title(
        "SPR_BENCH – Validation Complexity-Weighted Accuracy\n(batch-size comparison)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_val_cpxwa_vs_epoch.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA curve: {e}")
    plt.close()

# ------------ Plot 2: Train loss vs epoch ----------
try:
    plt.figure()
    for idx, bs in enumerate(b_sizes):
        plt.plot(epochs_ls[idx], train_loss[idx], marker="o", label=f"bs={bs}")
    plt.title("SPR_BENCH – Training Loss\n(batch-size comparison)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_train_loss_vs_epoch.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------ Plot 3: Final Val CpxWA vs batch size ----------
try:
    final_cpx = [vals[-1] for vals in val_cpx]
    plt.figure()
    plt.bar([str(bs) for bs in b_sizes], final_cpx, color="skyblue")
    plt.title(
        "SPR_BENCH – Final Validation Complexity-Weighted Accuracy\nper Batch Size"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Final CpxWA")
    fname = os.path.join(working_dir, "spr_bench_final_cpxwa_bar.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final bar chart: {e}")
    plt.close()

# ------------ Plot 4: Best-bs detailed CWA & SWA ----------
try:
    best_idx = int(np.argmax([vals[-1] for vals in val_cpx]))
    best_bs = b_sizes[best_idx]
    plt.figure()
    plt.plot(epochs_ls[best_idx], val_cwa[best_idx], marker="o", label="Color WA")
    plt.plot(epochs_ls[best_idx], val_swa[best_idx], marker="s", label="Shape WA")
    plt.title(f"SPR_BENCH – Val Weighted Accuracies for Best Batch Size (bs={best_bs})")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, f"spr_bench_best_bs{best_bs}_cwa_swa.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA/SWA plot: {e}")
    plt.close()

# ------------ Print final metrics ----------
for bs, cpx in zip(b_sizes, [v[-1] for v in val_cpx]):
    print(f"Batch Size {bs:>4}: Final Val CpxWA = {cpx:.4f}")
