import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ds_key = "SPR_BENCH"
    runs = experiment_data["dropout_rate"][ds_key]
    dropout_vals = sorted(runs.keys(), key=float)  # e.g. ['0.0','0.2',...]
    epochs = range(1, len(next(iter(runs.values()))["losses"]["train"]) + 1)

    # -------------- figure 1: loss curves -----------------
    try:
        plt.figure(figsize=(7, 4))
        for p in dropout_vals:
            train_loss = runs[p]["losses"]["train"]
            val_loss = runs[p]["losses"]["val"]
            plt.plot(
                epochs, train_loss, linestyle="--", marker="o", label=f"train p={p}"
            )
            plt.plot(epochs, val_loss, linestyle="-", marker="s", label=f"val p={p}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Training & Validation Loss vs Epoch")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss figure: {e}")
        plt.close()

    # -------------- figure 2: validation HWA -----------------
    try:
        plt.figure(figsize=(7, 4))
        for p in dropout_vals:
            hwa = [m["hwa"] for m in runs[p]["metrics"]["val"]]
            plt.plot(epochs, hwa, marker="^", label=f"p={p}")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR_BENCH Validation Harmonic Weighted Accuracy vs Epoch")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA figure: {e}")
        plt.close()

    # -------------- figure 3: final test HWA bar chart -----------------
    try:
        test_hwa = [runs[p]["metrics"]["test"]["hwa"] for p in dropout_vals]
        plt.figure(figsize=(6, 4))
        plt.bar(dropout_vals, test_hwa, color="skyblue")
        plt.xlabel("Dropout rate")
        plt.ylabel("Test HWA")
        plt.title("SPR_BENCH Final Test HWA per Dropout Rate")
        fname = os.path.join(working_dir, "SPR_BENCH_test_HWA_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()

    # -------------- print evaluation summary -----------------
    print("\n=== Final Test HWA Scores ===")
    for p, h in zip(dropout_vals, test_hwa):
        print(f"dropout {p}: HWA={h:.3f}")
