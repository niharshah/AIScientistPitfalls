import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

saved_files = []

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["batch_size"]["SPR_BENCH"]
    hyperparams = ed["hyperparams"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    hyperparams = []

# ------------- per-batch-size loss plots -------------
for idx, bs in enumerate(hyperparams[:5]):  # ensure ≤4 here
    try:
        train_l = ed["losses"]["train"][idx]
        val_l = ed["losses"]["val"][idx]
        epochs = np.arange(1, len(train_l) + 1)
        plt.figure()
        plt.plot(epochs, train_l, label="Train Loss")
        plt.plot(epochs, val_l, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title(f"SPR_BENCH Loss Curves - Batch Size {bs}")
        plt.legend()
        fname = f"SPR_BENCH_bs{bs}_loss.png"
        plt.savefig(os.path.join(working_dir, fname))
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for bs={bs}: {e}")
        plt.close()

# ------------- aggregated CompWA plot -------------
try:
    plt.figure()
    for idx, bs in enumerate(hyperparams[:5]):
        val_cwa = ed["metrics"]["val_CompWA"][idx]
        epochs = np.arange(1, len(val_cwa) + 1)
        plt.plot(epochs, val_cwa, label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.title("SPR_BENCH Validation Complexity-Weighted Accuracy")
    plt.legend()
    fname = "SPR_BENCH_val_CompWA_aggregated.png"
    plt.savefig(os.path.join(working_dir, fname))
    saved_files.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CompWA plot: {e}")
    plt.close()

print("Saved figures:", saved_files)
