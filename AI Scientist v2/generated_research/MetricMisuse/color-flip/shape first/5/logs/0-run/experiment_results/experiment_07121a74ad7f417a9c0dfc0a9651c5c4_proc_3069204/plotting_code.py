import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
# Load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
    decays = ed["decays"]
    train_loss_lists = ed["losses"]["train"]  # list of lists
    train_hsca_lists = ed["metrics"]["train"]  # list of lists
    test_hsca_vals = ed["metrics"]["val"]  # list of floats

    epochs = range(1, len(train_loss_lists[0]) + 1)  # assumed equal length

    # ----------------------------------------------------------- Plot 1
    try:
        plt.figure()
        for d, loss_curve in zip(decays, train_loss_lists):
            plt.plot(epochs, loss_curve, marker="o", label=f"decay={d}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Train Loss Curves – SPR_BENCH\nWeight Decay Sweep")
        plt.legend()
        fpath = os.path.join(working_dir, "SPR_BENCH_train_loss_weight_decay.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating train-loss plot: {e}")
        plt.close()

    # ----------------------------------------------------------- Plot 2
    try:
        plt.figure()
        for d, metric_curve in zip(decays, train_hsca_lists):
            plt.plot(epochs, metric_curve, marker="s", label=f"decay={d}")
        plt.xlabel("Epoch")
        plt.ylabel("HSCA")
        plt.title("Train HSCA Curves – SPR_BENCH\nWeight Decay Sweep")
        plt.legend()
        fpath = os.path.join(working_dir, "SPR_BENCH_train_HSCA_weight_decay.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating train-HSCA plot: {e}")
        plt.close()

    # ----------------------------------------------------------- Plot 3
    try:
        plt.figure()
        plt.bar([str(d) for d in decays], test_hsca_vals, color="skyblue")
        plt.xlabel("Weight Decay")
        plt.ylabel("Test HSCA")
        plt.title("Test HSCA vs Weight Decay – SPR_BENCH")
        fpath = os.path.join(working_dir, "SPR_BENCH_test_HSCA_vs_decay.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating test-HSCA plot: {e}")
        plt.close()

    # ----------------------------------------------------------- Print summary
    print("Weight Decay  |  Test HSCA")
    for d, h in zip(decays, test_hsca_vals):
        print(f"{d:<12} :  {h:.4f}")
