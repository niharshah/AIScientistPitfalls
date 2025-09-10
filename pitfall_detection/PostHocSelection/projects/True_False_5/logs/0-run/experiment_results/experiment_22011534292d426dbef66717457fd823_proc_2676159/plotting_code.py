import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------- paths ----------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------- load -----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# ------------------ fetch dataset ------------------
try:
    runs = experiment_data["embedding_dim_tuning"]["SPR_BENCH"]
except KeyError:
    print("SPR_BENCH data not found in experiment_data")
    exit()

dims = sorted(runs.keys(), key=lambda x: int(x.split("_")[-1]))
epochs = np.arange(1, len(runs[dims[0]]["losses"]["train"]) + 1)

# ------------------ figure 1: loss curves ----------
try:
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for k in dims:
        ax1.plot(epochs, runs[k]["losses"]["train"], label=k)
        ax2.plot(epochs, runs[k]["losses"]["val"], label=k)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation")
    plt.suptitle(
        "SPR_BENCH Loss Curves (Embedding Dim Comparison)\nLeft: Training Loss, Right: Validation Loss"
    )
    ax1.legend()
    ax2.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------ figure 2: RCWA curves ----------
try:
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for k in dims:
        ax1.plot(epochs, runs[k]["metrics"]["train"], label=k)
        ax2.plot(epochs, runs[k]["metrics"]["val"], label=k)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("RCWA")
    ax1.set_title("Train")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RCWA")
    ax2.set_title("Validation")
    plt.suptitle(
        "SPR_BENCH RCWA Curves (Embedding Dim Comparison)\nLeft: Training RCWA, Right: Validation RCWA"
    )
    ax1.legend()
    ax2.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_RCWA_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating RCWA curves: {e}")
    plt.close()

# -------- figure 3: Test RCWA bar ------------------
try:
    rcwa_vals = [runs[k]["test_metrics"]["RCWA"] for k in dims]
    plt.figure(figsize=(6, 4))
    plt.bar(dims, rcwa_vals, color="skyblue")
    plt.ylabel("RCWA")
    plt.xlabel("Embedding Dim")
    plt.title("SPR_BENCH Test RCWA vs Embedding Dim")
    fname = os.path.join(working_dir, "SPR_BENCH_test_RCWA.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test RCWA bar: {e}")
    plt.close()

# -------- figure 4: Test SWA & CWA bar -------------
try:
    swa_vals = [runs[k]["test_metrics"]["SWA"] for k in dims]
    cwa_vals = [runs[k]["test_metrics"]["CWA"] for k in dims]
    x = np.arange(len(dims))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, swa_vals, width, label="SWA")
    plt.bar(x + width / 2, cwa_vals, width, label="CWA")
    plt.xticks(x, dims)
    plt.ylabel("Score")
    plt.xlabel("Embedding Dim")
    plt.title("SPR_BENCH Test SWA and CWA vs Embedding Dim")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_SWA_CWA.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating SWA/CWA bar: {e}")
    plt.close()
