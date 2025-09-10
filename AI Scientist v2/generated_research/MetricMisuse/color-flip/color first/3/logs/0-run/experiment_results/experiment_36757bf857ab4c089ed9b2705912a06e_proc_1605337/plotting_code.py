import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# 1. Load experiment data                                            #
# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bs_logs = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})

# ------------------------------------------------------------------ #
# 2. Per-batch-size training curves (≤5 figures)                     #
# ------------------------------------------------------------------ #
for bs_idx, (bs, log) in enumerate(sorted(bs_logs.items())):
    try:
        # Extract epoch-wise data
        epochs = [e for e, _ in log["losses"]["train"]]
        train_loss = [l for _, l in log["losses"]["train"]]
        val_loss = [l for _, l in log["losses"]["val"]]
        val_hcsa = [h for _, _, _, h in log["metrics"]["val"]]

        # Build figure with two subplots
        plt.figure(figsize=(10, 4))

        # Left: losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("CE Loss")
        plt.title("Loss Curves")
        plt.legend()

        # Right: HCSA
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_hcsa, marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("HCSA")
        plt.title("Validation HCSA")

        plt.suptitle(f"SPR_BENCH Batch Size {bs}\nLeft: Loss, Right: HCSA")
        fname = os.path.join(working_dir, f"SPR_BENCH_bs{bs}_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating curve plot for bs={bs}: {e}")
        plt.close()

# ------------------------------------------------------------------ #
# 3. Summary bar chart of final test HCSA                            #
# ------------------------------------------------------------------ #
try:
    plt.figure()
    bss = sorted(bs_logs.keys())
    hcsa_vals = [bs_logs[b]["final_test_metrics"][2] for b in bss]
    plt.bar([str(b) for b in bss], hcsa_vals, color="skyblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Test HCSA")
    plt.title("SPR_BENCH Final Test HCSA by Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_HCSA_by_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating summary HCSA plot: {e}")
    plt.close()

print("Plots saved to:", working_dir)
