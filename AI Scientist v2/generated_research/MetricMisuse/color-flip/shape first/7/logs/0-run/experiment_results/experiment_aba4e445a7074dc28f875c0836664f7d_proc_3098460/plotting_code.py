import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths & load ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    runs = experiment_data["hidden_dim"]["SPR_BENCH"]
    hidden_sizes = sorted(int(h) for h in runs.keys())

    # --------- PLOT 1: loss curves -----------
    try:
        plt.figure(figsize=(6, 4))
        for h in hidden_sizes:
            run = runs[str(h)]
            epochs, tr_loss = zip(*run["losses"]["train"])
            _, val_loss = zip(*run["losses"]["val"])
            plt.plot(epochs, tr_loss, label=f"{h}-train", linestyle="--")
            plt.plot(epochs, val_loss, label=f"{h}-val")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # --------- PLOT 2: HWA curves ------------
    try:
        plt.figure(figsize=(6, 4))
        for h in hidden_sizes:
            run = runs[str(h)]
            epochs, swa, cwa, hwa = zip(*run["metrics"]["val"])
            plt.plot(epochs, hwa, label=f"{h}-HWA")
        plt.title("SPR_BENCH: Harmonic-Weighted Accuracy (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve plot: {e}")
        plt.close()

    # --------- PLOT 3: final HWA vs hidden ----
    try:
        final_hwa = []
        for h in hidden_sizes:
            hwa_last = runs[str(h)]["metrics"]["val"][-1][3]
            final_hwa.append(hwa_last)
        plt.figure(figsize=(5, 3))
        plt.bar([str(h) for h in hidden_sizes], final_hwa, color="skyblue")
        plt.title("SPR_BENCH: Final Epoch HWA vs Hidden Size")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Final Harmonic-Weighted Accuracy")
        fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating final HWA bar plot: {e}")
        plt.close()

    # --------- Print summary -----------------
    print("\nFinal epoch validation HWA by hidden size:")
    for h, hwa in zip(hidden_sizes, final_hwa):
        print(f"  hidden={h:3d} -> HWA = {hwa:.4f}")
