import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------------- #
# 1. setup & load                                             #
# ----------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def unpack(list_of_tuples, idx):
    return [t[idx] for t in list_of_tuples]


try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = exp.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = {}

# ----------------------------------------------------------- #
# 2. Train / Val loss curve                                   #
# ----------------------------------------------------------- #
try:
    plt.figure()
    ep_tr = unpack(run["losses"]["train"], 0)
    tr_loss = unpack(run["losses"]["train"], 1)
    ep_val = unpack(run["losses"]["val"], 0)
    val_loss = unpack(run["losses"]["val"], 1)
    plt.plot(ep_tr, tr_loss, "--", label="Train")
    plt.plot(ep_val, val_loss, "-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title("SPR_BENCH: Train vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 3. Validation HCSA curve                                    #
# ----------------------------------------------------------- #
try:
    plt.figure()
    ep_val = unpack(run["metrics"]["val"], 0)
    hcs = [t[3] for t in run["metrics"]["val"]]
    plt.plot(ep_val, hcs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("HCSA")
    plt.title("SPR_BENCH: Validation HCSA")
    fname = os.path.join(working_dir, "SPR_BENCH_val_HCSA_curve.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating HCSA curve plot: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 4. Best HCSA bar chart                                      #
# ----------------------------------------------------------- #
try:
    best_hcs = max(hcs) if hcs else 0
    best_ep = ep_val[int(np.argmax(hcs))] if hcs else -1
    plt.figure()
    plt.bar(["SPR_BENCH"], [best_hcs])
    plt.ylabel("Best Validation HCSA")
    plt.title(f"SPR_BENCH: Best HCSA (epoch {best_ep})")
    fname = os.path.join(working_dir, "SPR_BENCH_best_HCSA_bar.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating best HCSA bar chart: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 5. Validation CWA vs SWA                                    #
# ----------------------------------------------------------- #
try:
    plt.figure()
    cwa = [t[1] for t in run["metrics"]["val"]]
    swa = [t[2] for t in run["metrics"]["val"]]
    plt.plot(ep_val, cwa, label="CWA")
    plt.plot(ep_val, swa, label="SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation CWA and SWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_CWA_SWA.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating CWA/SWA plot: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 6. Text summary                                             #
# ----------------------------------------------------------- #
if hcs:
    print(f"\nBest validation HCSA = {best_hcs:.3f} at epoch {best_ep}")
