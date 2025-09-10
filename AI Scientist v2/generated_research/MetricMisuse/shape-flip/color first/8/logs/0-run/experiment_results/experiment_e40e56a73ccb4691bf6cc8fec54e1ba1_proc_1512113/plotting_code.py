import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    tags = list(experiment_data.get("batch_size", {}).keys())

    # ---------- 1. validation loss curves ----------
    try:
        plt.figure()
        for tag in tags:
            vals = experiment_data["batch_size"][tag]["losses"]["val"]
            plt.plot(vals, label=tag)
        plt.title("Validation Cross-Entropy vs Epoch (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot: {e}")
        plt.close()

    # ---------- 2. accuracy curves ----------
    try:
        fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(10, 4))
        for tag in tags:
            tr = experiment_data["batch_size"][tag]["metrics"]["train"]
            val = experiment_data["batch_size"][tag]["metrics"]["val"]
            ax_tr.plot(tr, label=tag)
            ax_val.plot(val, label=tag)
        ax_tr.set_title("Left: Training Accuracy")
        ax_val.set_title("Right: Validation Accuracy")
        for ax in (ax_tr, ax_val):
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
        fig.suptitle("Accuracy vs Epoch (SPR_BENCH)")
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve plot: {e}")
        plt.close()

    # ---------- 3. final validation accuracy bar ----------
    try:
        plt.figure()
        final_acc = [
            experiment_data["batch_size"][tag]["metrics"]["val"][-1] for tag in tags
        ]
        plt.bar(tags, final_acc)
        plt.title("Final Validation Accuracy by Batch Size (SPR_BENCH)")
        plt.xlabel("Batch Size Setting")
        plt.ylabel("Accuracy")
        fname = os.path.join(working_dir, "SPR_BENCH_final_val_acc.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final accuracy bar plot: {e}")
        plt.close()
