import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------- #
#  load experiment data
# ------------------------------------------------- #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    # collect final test SWA for bar chart
    bar_names, bar_vals = [], []

    for dset, run in exp.items():
        # ---------------- Loss curves ----------------
        try:
            tr_loss = run["losses"].get("train")
            val_loss = run["losses"].get("val")
            if tr_loss and val_loss:
                x = np.arange(len(tr_loss))
                plt.figure(figsize=(6, 4))
                plt.plot(x, tr_loss, "--", label="Train")
                plt.plot(x, val_loss, "-", label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{dset} Loss Curves\nTrain (dashed) vs Validation (solid)")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                print(f"Saved {fname}")
                plt.close()
        except Exception as e:
            print(f"Error plotting loss for {dset}: {e}")
            plt.close()

        # ----------- Metric (SWA) curves -------------
        try:
            tr_swa = run["metrics"].get("train")
            val_swa = run["metrics"].get("val")
            if tr_swa and val_swa:
                # sample at most 50 points
                step = max(1, len(val_swa) // 50)
                x = np.arange(len(val_swa))[::step]
                plt.figure(figsize=(6, 4))
                plt.plot(x, np.array(tr_swa)[::step], "--", label="Train")
                plt.plot(x, np.array(val_swa)[::step], "-", label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Shape-Weighted Accuracy")
                plt.title(f"{dset} SWA Curves\nTrain (dashed) vs Validation (solid)")
                plt.legend()
                fname = os.path.join(working_dir, f"{dset}_swa_curves.png")
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                print(f"Saved {fname}")
                plt.close()
        except Exception as e:
            print(f"Error plotting metric for {dset}: {e}")
            plt.close()

        # gather test SWA
        test_swa = run["metrics"].get("test")
        if test_swa is not None:
            bar_names.append(dset)
            bar_vals.append(test_swa)
            print(f"Final Test SWA for {dset}: {test_swa:.4f}")

    # ----------- Bar chart across datasets ----------
    try:
        if bar_names:
            plt.figure(figsize=(6, 4))
            plt.bar(bar_names, bar_vals, color="skyblue")
            plt.ylabel("Final Test Shape-Weighted Accuracy")
            plt.title("Comparison of Test SWA Across Datasets")
            fname = os.path.join(working_dir, "all_datasets_test_swa_bar.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()
