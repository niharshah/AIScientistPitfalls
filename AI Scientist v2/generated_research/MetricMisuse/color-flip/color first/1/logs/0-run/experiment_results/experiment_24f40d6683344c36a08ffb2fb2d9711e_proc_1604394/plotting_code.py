import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
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
    # gather summaries
    epochs_dict, train_loss, val_loss, val_hmwa, test_hmwa = {}, {}, {}, {}, {}
    for tag in tags:
        ed = experiment_data[tag]["SPR_BENCH"]
        train_loss[tag] = ed["losses"]["train"]
        val_loss[tag] = ed["losses"]["val"]
        val_hmwa[tag] = [m["hmwa"] for m in ed["metrics"]["val"]]
        test_hmwa[tag] = ed["metrics"]["test"]["hmwa"]
        epochs_dict[tag] = list(range(1, len(train_loss[tag]) + 1))

    # ---------------- plot 1 : Loss curves ----------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
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
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------------- plot 2 : Validation HMWA ----------------
    try:
        plt.figure(figsize=(6, 4))
        for tag in tags:
            plt.plot(epochs_dict[tag], val_hmwa[tag], label=tag)
        plt.title("SPR_BENCH Validation HMWA over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("HMWA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_HMWA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HMWA plot: {e}")
        plt.close()

    # ---------------- plot 3 : Test HMWA bar ----------------
    try:
        plt.figure(figsize=(6, 4))
        names, scores = zip(*sorted(test_hmwa.items()))
        plt.bar(names, scores, color="skyblue")
        plt.title("SPR_BENCH Test HMWA by Hidden Dimension")
        plt.ylabel("HMWA")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_HMWA_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test HMWA bar plot: {e}")
        plt.close()

    # -------- print final test metrics --------
    print("\nTest-set performance:")
    for tag in tags:
        met = experiment_data[tag]["SPR_BENCH"]["metrics"]["test"]
        print(
            f"{tag}: CWA={met['cwa']:.4f}, SWA={met['swa']:.4f}, HMWA={met['hmwa']:.4f}"
        )
