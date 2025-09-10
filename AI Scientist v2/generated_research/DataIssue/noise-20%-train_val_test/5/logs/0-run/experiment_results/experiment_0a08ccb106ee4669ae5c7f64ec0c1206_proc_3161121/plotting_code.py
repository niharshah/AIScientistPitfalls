import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load data -----
try:
    exp_file = os.path.join(working_dir, "experiment_data.npy")
    if not os.path.isfile(exp_file):  # fallback to current dir if not moved
        exp_file = "experiment_data.npy"
    experiment_data = np.load(exp_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data["embed_dim_tuning"]["SPR_BENCH"]["runs"]
    embed_dims = [r["embed_dim"] for r in runs]
    # ---------- plot train/val accuracy ----------
    try:
        plt.figure()
        for r in runs:
            epochs = r["epoch"]
            plt.plot(
                epochs, r["metrics"]["train_acc"], label=f"train ed={r['embed_dim']}"
            )
            plt.plot(
                epochs, r["metrics"]["val_acc"], "--", label=f"val ed={r['embed_dim']}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs. Validation Accuracy (Embed Dim Tuning)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_acc_curves_embed_dim.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- plot train/val loss ----------
    try:
        plt.figure()
        for r in runs:
            epochs = r["epoch"]
            plt.plot(epochs, r["losses"]["train"], label=f"train ed={r['embed_dim']}")
            plt.plot(epochs, r["losses"]["val"], "--", label=f"val ed={r['embed_dim']}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs. Validation Loss (Embed Dim Tuning)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves_embed_dim.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- bar chart of test accuracy ----------
    try:
        plt.figure()
        test_accs = [r["test_acc"] for r in runs]
        plt.bar(range(len(embed_dims)), test_accs, tick_label=embed_dims)
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Test Accuracy")
        plt.title("SPR_BENCH: Test Accuracy per Embedding Dimension")
        fname = os.path.join(working_dir, "spr_bench_test_acc_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar plot: {e}")
        plt.close()

    # ---------- print test accuracies ----------
    for ed, ta in zip(embed_dims, test_accs):
        print(f"Embed dim {ed}: test_acc = {ta:.4f}")
