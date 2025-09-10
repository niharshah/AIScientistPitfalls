import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("embed_dim_tuning", {})
dims_sorted = sorted(runs.keys(), key=lambda x: int(x.split("_")[-1]))[:5]  # keep order

# ------------------- per-run plots -------------------
for run_key in dims_sorted:
    try:
        data = runs[run_key]
        epochs = range(1, len(data["losses"]["train"]) + 1)
        train_loss = data["losses"]["train"]
        val_loss = data["losses"]["val"]
        val_crwa = [m["CRWA"] for m in data["metrics"]["val"]]

        plt.figure(figsize=(10, 4))
        # left panel: loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")

        # right panel: CRWA
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_crwa, marker="o", label="Val CRWA")
        plt.xlabel("Epoch")
        plt.ylabel("CRWA")
        plt.legend()
        plt.title("CRWA Over Epochs")

        dim = run_key.split("_")[-1]
        plt.suptitle(
            f"SPR_BENCH | Embed Dim {dim}\nLeft: Loss | Right: CRWA", fontsize=12
        )
        fname = os.path.join(working_dir, f"SPR_BENCH_dim_{dim}_loss_crwa.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating plot for {run_key}: {e}")
        plt.close()

# ------------------- summary bar chart -------------------
try:
    dims, crwas = [], []
    for run_key in dims_sorted:
        dims.append(int(run_key.split("_")[-1]))
        crwas.append(runs[run_key]["metrics"]["test"]["CRWA"])
    plt.figure()
    plt.bar([str(d) for d in dims], crwas, color="skyblue")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Test CRWA")
    plt.title("SPR_BENCH | Test CRWA vs Embedding Dimension")
    fname = os.path.join(working_dir, "SPR_BENCH_test_CRWA_vs_dim.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()
