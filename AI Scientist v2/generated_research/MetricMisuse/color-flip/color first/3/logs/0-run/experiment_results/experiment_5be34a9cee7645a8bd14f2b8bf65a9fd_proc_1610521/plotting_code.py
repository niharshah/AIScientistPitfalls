import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------- setup --------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------- load ---------------------------------------- #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}


# helper
def unpack(tuples, idx):
    return [t[idx] for t in tuples]


# ----------------------------- iterate datasets ---------------------------- #
for dset, blob in exp.items():
    print(f"\nProcessing dataset: {dset}")
    losses = blob.get("losses", {})
    metrics = blob.get("metrics", {})
    # --------------------- 1. loss curves ---------------------------------- #
    try:
        plt.figure()
        if losses.get("train"):
            plt.plot(
                unpack(losses["train"], 0),
                unpack(losses["train"], 1),
                "--",
                label="train",
            )
        if losses.get("val"):
            plt.plot(
                unpack(losses["val"], 0), unpack(losses["val"], 1), "-", label="val"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title(f"{dset}: Train vs. Val Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
        plt.savefig(fname, dpi=150)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
    finally:
        plt.close()

    # --------------------- 2. validation metric curves --------------------- #
    try:
        vals = metrics.get("val", [])
        if vals:
            epochs = unpack(vals, 0)
            cwa_vals = unpack(vals, 1)
            swa_vals = unpack(vals, 2)
            snwa_vals = unpack(vals, 3)
            plt.figure()
            plt.plot(epochs, cwa_vals, label="CWA")
            plt.plot(epochs, swa_vals, label="SWA")
            plt.plot(epochs, snwa_vals, label="SNWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dset}: Validation Metrics over Epochs")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_val_metric_curves.png")
            plt.savefig(fname, dpi=150)
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating metric curves for {dset}: {e}")
    finally:
        plt.close()

    # --------------------- 3. best SNWA bar chart -------------------------- #
    try:
        vals = metrics.get("val", [])
        if vals:
            best_snwa = max(unpack(vals, 3))
            plt.figure()
            plt.bar([0], [best_snwa], tick_label=[dset])
            plt.ylabel("Best SNWA")
            plt.title(f"{dset}: Best Validation SNWA")
            fname = os.path.join(working_dir, f"{dset}_best_SNWA_bar.png")
            plt.savefig(fname, dpi=150)
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating best SNWA bar for {dset}: {e}")
    finally:
        plt.close()

    # --------------------- 4. test accuracy print -------------------------- #
    try:
        preds = blob.get("predictions", {}).get("test", [])
        gts = blob.get("ground_truth", {}).get("test", [])
        if preds and gts:
            acc = sum(int(p == g) for p, g in zip(preds, gts)) / len(gts)
            print(f"{dset}: Test accuracy = {acc:.3%} on {len(gts)} samples")
    except Exception as e:
        print(f"Error computing test accuracy for {dset}: {e}")
