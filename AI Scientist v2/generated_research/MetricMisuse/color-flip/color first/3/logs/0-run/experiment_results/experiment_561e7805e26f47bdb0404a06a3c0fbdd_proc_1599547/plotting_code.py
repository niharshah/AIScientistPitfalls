import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper: safely fetch data
def get_list(d, *keys):
    cur = d
    for k in keys:
        if k not in cur:
            return []
        cur = cur[k]
    return cur


# Plot 1: Loss curves ----------------------------------------------------------
try:
    train_losses = get_list(experiment_data, "SPR_BENCH", "losses", "train")
    val_losses = get_list(experiment_data, "SPR_BENCH", "losses", "val")
    if train_losses and val_losses:
        epochs_t, loss_t = zip(*train_losses)
        epochs_v, loss_v = zip(*val_losses)
        plt.figure()
        plt.plot(epochs_t, loss_t, label="Train")
        plt.plot(epochs_v, loss_v, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Training, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot 2: Validation HCSA ------------------------------------------------------
try:
    val_metrics = get_list(experiment_data, "SPR_BENCH", "metrics", "val")
    if val_metrics:
        epochs_m, hcsas = zip(*val_metrics)
        plt.figure()
        plt.plot(epochs_m, hcsas, marker="o")
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic CSA")
        plt.title("SPR_BENCH Validation Harmonic CSA\nDataset: SPR_BENCH dev split")
        fname = os.path.join(working_dir, "SPR_BENCH_HCSA_val.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating HCSA curve: {e}")
    plt.close()

# Plot 3: Test HCSA ------------------------------------------------------------
try:
    hcs_test = get_list(experiment_data, "SPR_BENCH", "metrics", "test")
    if isinstance(hcs_test, (int, float)):
        plt.figure()
        plt.bar(["HCSA"], [hcs_test])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Test Harmonic CSA\nDataset: SPR_BENCH test split")
        fname = os.path.join(working_dir, "SPR_BENCH_HCSA_test.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test HCSA bar: {e}")
    plt.close()

# --------- print main metric -------------
if isinstance(get_list(experiment_data, "SPR_BENCH", "metrics", "test"), (int, float)):
    print(f"Test HCSA: {experiment_data['SPR_BENCH']['metrics']['test']:.3f}")
