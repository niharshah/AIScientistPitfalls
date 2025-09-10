import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick safety check
results = (
    experiment_data.get("dropout_tuning", {}).get("SPR_BENCH", {}).get("results", [])
)
if not results:
    print("No results to plot.")
else:
    # ------------ 1) F1 curves ------------
    try:
        plt.figure(figsize=(10, 4))
        # left subplot: train F1
        ax1 = plt.subplot(1, 2, 1)
        for r in results:
            ax1.plot(r["epochs"], r["metrics"]["train"], label=f"dp={r['dropout']}")
        ax1.set_title("Train F1")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Macro-F1")
        ax1.legend()

        # right subplot: val F1
        ax2 = plt.subplot(1, 2, 2)
        for r in results:
            ax2.plot(r["epochs"], r["metrics"]["val"], label=f"dp={r['dropout']}")
        ax2.set_title("Validation F1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro-F1")
        ax2.legend()

        plt.suptitle("SPR_BENCH – Left: Training, Right: Validation F1 Curves")
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # ------------ 2) Loss curves ------------
    try:
        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        for r in results:
            ax1.plot(r["epochs"], r["losses"]["train"], label=f"dp={r['dropout']}")
        ax1.set_title("Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("CE Loss")
        ax1.legend()

        ax2 = plt.subplot(1, 2, 2)
        for r in results:
            ax2.plot(r["epochs"], r["losses"]["val"], label=f"dp={r['dropout']}")
        ax2.set_title("Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("CE Loss")
        ax2.legend()

        plt.suptitle("SPR_BENCH – Left: Training, Right: Validation Loss Curves")
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------ 3) Test macro-F1 vs dropout ------------
    try:
        plt.figure(figsize=(6, 4))
        dps = [r["dropout"] for r in results]
        test_f1 = [r["test_macroF1"] for r in results]
        plt.bar([str(dp) for dp in dps], test_f1, color="skyblue")
        plt.xlabel("Dropout")
        plt.ylabel("Test Macro-F1")
        plt.title("SPR_BENCH – Test Macro-F1 vs Dropout")
        fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test F1 bar plot: {e}")
        plt.close()

    # ----------- print evaluation metrics ------------
    print("\n=== Test Macro-F1 Scores ===")
    for r in results:
        print(f"Dropout {r['dropout']:.2f}: {r['test_macroF1']:.4f}")
