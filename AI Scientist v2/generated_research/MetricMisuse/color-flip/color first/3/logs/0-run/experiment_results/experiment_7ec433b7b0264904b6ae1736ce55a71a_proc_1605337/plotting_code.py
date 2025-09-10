import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------- #
# 1. Load experiment data                                     #
# ----------------------------------------------------------- #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = exp["epochs_tuning"]["SPR_BENCH"]["runs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}


# helper to fetch data safely
def unpack(list_of_tuples, idx):
    return [t[idx] for t in list_of_tuples]


# ----------------------------------------------------------- #
# 2. Plot: train / val loss curves                            #
# ----------------------------------------------------------- #
try:
    plt.figure()
    for name, run in runs.items():
        tr_epochs = unpack(run["losses"]["train"], 0)
        tr_loss = unpack(run["losses"]["train"], 1)
        val_epochs = unpack(run["losses"]["val"], 0)
        val_loss = unpack(run["losses"]["val"], 1)
        plt.plot(tr_epochs, tr_loss, "--", label=f"{name}-train")
        plt.plot(val_epochs, val_loss, "-", label=f"{name}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("SPR_BENCH: Train vs. Val Loss")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 3. Plot: validation HCSA curves                             #
# ----------------------------------------------------------- #
try:
    plt.figure()
    for name, run in runs.items():
        val_epochs = unpack(run["metrics"]["val"], 0)
        hcs = [t[3] for t in run["metrics"]["val"]]
        plt.plot(val_epochs, hcs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("HCSA")
    plt.title("SPR_BENCH: Validation HCSA")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_val_HCSA_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating HCSA curve plot: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 4. Plot: best HCSA per run (bar-chart)                      #
# ----------------------------------------------------------- #
best_vals, labels = [], []
for name, run in runs.items():
    hcs_list = [t[3] for t in run["metrics"]["val"]]
    if hcs_list:
        best_vals.append(max(hcs_list))
        labels.append(name)

try:
    plt.figure()
    plt.bar(range(len(best_vals)), best_vals, tick_label=labels)
    plt.ylabel("Best Validation HCSA")
    plt.title("SPR_BENCH: Best HCSA vs. Epoch Budget")
    plt.xticks(rotation=45, ha="right")
    fname = os.path.join(working_dir, "SPR_BENCH_best_HCSA_bar.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating best HCSA bar chart: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 5. Print summary table                                      #
# ----------------------------------------------------------- #
print("\nSummary of best validation HCSA per run:")
for name, run in runs.items():
    hcs_list = [t[3] for t in run["metrics"]["val"]]
    ep_list = unpack(run["metrics"]["val"], 0)
    if hcs_list:
        best_idx = int(np.argmax(hcs_list))
        print(
            f"{name:>12}: best HCSA={hcs_list[best_idx]:.3f} at epoch {ep_list[best_idx]}"
        )
