import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ LOAD EXPERIMENT DATA ------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Quick exit if nothing to plot
if not experiment_data:
    exit()

results_dict = (
    experiment_data.get("NoPaddingMask", {}).get("SPR_BENCH", {}).get("results", {})
)
nheads = sorted(results_dict.keys(), key=int)

# ------------------ PER-NHEAD ACCURACY CURVES --------------- #
for nh in nheads:
    try:
        metrics = results_dict[nh]["metrics"]
        train_acc = metrics["train_acc"]
        val_acc = metrics["val_acc"]
        epochs = np.arange(1, len(train_acc) + 1)

        plt.figure()
        plt.plot(epochs, train_acc, marker="o", label="Train Acc")
        plt.plot(epochs, val_acc, marker="s", label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"SPR_BENCH – NoPaddingMask / Train vs Val Acc\nnhead={nh}")
        plt.legend()
        fname = f"SPR_BENCH_nhead{nh}_train_val_acc.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating acc curve for nhead={nh}: {e}")
        plt.close()

# ------------------ TEST ACCURACY BARPLOT ------------------ #
try:
    test_accs = [results_dict[nh]["test_acc"] for nh in nheads]
    plt.figure()
    plt.bar(range(len(nheads)), test_accs, tick_label=nheads)
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH – NoPaddingMask / Test Acc vs nhead")
    for idx, acc in enumerate(test_accs):
        plt.text(idx, acc + 0.01, f"{acc:.2f}", ha="center")
    fname = "SPR_BENCH_test_acc_barplot.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()

print("Finished plotting experiment results.")
