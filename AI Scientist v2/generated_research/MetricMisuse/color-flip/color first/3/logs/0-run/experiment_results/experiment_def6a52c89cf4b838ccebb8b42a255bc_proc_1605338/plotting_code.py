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
    runs = experiment_data.get("weight_decay", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}


# ------------------------------------------------------------------ #
# Helper to collect series                                           #
# ------------------------------------------------------------------ #
def collect_series(key_chain, run):
    """key_chain e.g. ('losses','train') returns epochs,list(vals)."""
    tup_list = run
    for k in key_chain:
        tup_list = tup_list[k]
    epochs, vals = zip(*tup_list)
    return np.array(epochs), np.array(vals)


# ------------------------------------------------------------------ #
# Plot 1: Loss curves                                                #
# ------------------------------------------------------------------ #
try:
    plt.figure()
    for wd, run in runs.items():
        ep_t, tr_loss = collect_series(("losses", "train"), run)
        ep_v, val_loss = collect_series(("losses", "val"), run)
        plt.plot(ep_t, tr_loss, label=f"Train wd={wd}")
        plt.plot(ep_v, val_loss, "--", label=f"Val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nTrain vs Validation for Different Weight Decays")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Plot 2: Validation HCSA                                            #
# ------------------------------------------------------------------ #
try:
    plt.figure()
    for wd, run in runs.items():
        ep, metrics = zip(*run["metrics"]["val"])
        hcs = [m[2] for m in metrics]  # CWA, SWA, HCSA -> idx2
        plt.plot(ep, hcs, label=f"HCSA wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic CSA")
    plt.title("SPR_BENCH Validation HCSA\nEffect of Weight Decay")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "SPR_BENCH_val_HCSA.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating HCSA plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# Print best validation HCSA per weight decay                        #
# ------------------------------------------------------------------ #
for wd, run in runs.items():
    hcs = [m[2] for _, *m in run["metrics"]["val"]]
    best = max(hcs) if hcs else float("nan")
    print(f"Best Val HCSA for weight_decay={wd}: {best:.3f}")
