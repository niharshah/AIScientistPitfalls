import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set working directory ----------
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

# ---------- loss curves ----------
for idx, (run_name, run_data) in enumerate(experiment_data.items()):
    if idx >= 4:  # safety cap (guideline)
        break
    try:
        tr_loss = run_data.get("losses", {}).get("train", [])
        val_loss = run_data.get("losses", {}).get("val", [])
        if not tr_loss or not val_loss:
            raise ValueError("Missing loss data")
        epochs = np.arange(1, len(tr_loss) + 1)

        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{run_name}: Loss Curves\nDataset: SPR_BENCH (toy)", fontsize=10)
        plt.legend()
        fname = os.path.join(working_dir, f"{run_name}_loss_curve_SPR_BENCH.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {run_name}: {e}")
        plt.close()

# ---------- aggregated test SWA ----------
try:
    names, swa_vals = [], []
    for run_name, run_data in experiment_data.items():
        names.append(run_name)
        swa_vals.append(run_data.get("metrics", {}).get("test", 0))

    x = np.arange(len(names))
    plt.figure(figsize=(6, 4))
    plt.bar(x, swa_vals, width=0.4, color="steelblue")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("Final Test SWA Across Variants\nDataset: SPR_BENCH (toy)", fontsize=10)
    fname = os.path.join(working_dir, "test_SWA_comparison_SPR_BENCH.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated SWA plot: {e}")
    plt.close()

# ---------- validation SWA curves ----------
try:
    plt.figure()
    for idx, (run_name, run_data) in enumerate(experiment_data.items()):
        if idx >= 5:  # guideline cap
            break
        val_swa = run_data.get("metrics", {}).get("val", [])
        if not val_swa:
            continue
        epochs = np.arange(1, len(val_swa) + 1)
        plt.plot(epochs, val_swa, label=run_name)
    if plt.gca().has_data():
        plt.xlabel("Epoch")
        plt.ylabel("Validation SWA")
        plt.title(
            "Validation SWA vs Epochs (All Variants)\nDataset: SPR_BENCH (toy)",
            fontsize=10,
        )
        plt.legend()
        fname = os.path.join(working_dir, "val_SWA_curves_SPR_BENCH.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation SWA curve plot: {e}")
    plt.close()

# ---------- print evaluation metrics ----------
print("Final Test SWA:")
for run_name, run_data in experiment_data.items():
    swa = run_data.get("metrics", {}).get("test", 0)
    print(f"{run_name}: SWA={swa:.4f}")
