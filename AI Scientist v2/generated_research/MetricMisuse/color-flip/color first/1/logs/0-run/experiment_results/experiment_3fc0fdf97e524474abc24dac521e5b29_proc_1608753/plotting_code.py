import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------- paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------- load data dict
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = list(experiment_data.keys())
if not tags:
    print("No experiment data found, nothing to plot.")
else:
    # ----------------------------------------- gather per-tag series
    epochs_dict, train_loss, val_loss = {}, {}, {}
    val_cva, test_cva = {}, {}
    for tag in tags:
        ed = experiment_data[tag]["SPR_BENCH"]
        train_loss[tag] = ed["losses"]["train"]
        val_loss[tag] = ed["losses"]["val"]
        val_cva[tag] = [m["cva"] for m in ed["metrics"]["val"]]
        test_cva[tag] = ed["metrics"]["test"]["cva"]
        epochs_dict[tag] = list(range(1, len(train_loss[tag]) + 1))

    # ----------------------------------------- plot 1 : Loss curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for tag in tags:
            axes[0].plot(epochs_dict[tag], train_loss[tag], label=tag)
            axes[1].plot(epochs_dict[tag], val_loss[tag], label=tag)
        axes[0].set_title("Train Loss")
        axes[1].set_title("Validation Loss")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy")
            ax.legend()
        fig.suptitle("SPR_BENCH Loss Curves (Left: Train, Right: Validation)")
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ----------------------------------------- plot 2 : Validation CVA
    try:
        plt.figure(figsize=(6, 4))
        for tag in tags:
            plt.plot(epochs_dict[tag], val_cva[tag], label=tag)
        plt.title("SPR_BENCH Validation Composite-Variety Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("CVA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_CVA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CVA plot: {e}")
        plt.close()

    # ----------------------------------------- plot 3 : Test CVA bar
    try:
        plt.figure(figsize=(6, 4))
        names, scores = zip(*sorted(test_cva.items()))
        plt.bar(names, scores, color="steelblue")
        plt.title("SPR_BENCH Test Composite-Variety Accuracy")
        plt.ylabel("CVA")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_CVA_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test CVA bar plot: {e}")
        plt.close()

    # ----------------------------------------- console summary
    print("\n=== Test-set metrics ===")
    for tag in tags:
        m = experiment_data[tag]["SPR_BENCH"]["metrics"]["test"]
        print(f"{tag}: CWA={m['cwa']:.4f} | SWA={m['swa']:.4f} | CVA={m['cva']:.4f}")
