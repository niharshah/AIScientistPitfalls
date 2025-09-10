import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ setup & loading ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    for dset, rec in experiment_data.items():
        tr_losses = rec["losses"]["train"]
        val_losses = rec["losses"]["val"]
        test_metrics = rec["metrics"]["test"]
        # ---------- 1) learning curves ----------
        try:
            epochs = range(1, len(tr_losses) + 1)
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, tr_losses, "b-o", label="Train Loss")
            plt.plot(epochs, val_losses, "r-o", label="Val Loss")
            plt.title(f"{dset}: Loss Curves\nLeft: Train, Right: Val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dset}: {e}")
            plt.close()
        # ---------- 2) test SWA ----------
        try:
            swa = test_metrics.get("swa", None)
            if swa is not None:
                plt.figure(figsize=(3, 4))
                plt.bar(["SWA"], [swa], color="orange")
                plt.ylim(0, 1)
                plt.title(f"{dset}: Test Shape-Weighted Accuracy")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{dset}_test_swa.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
                plt.close()
            else:
                print(f"SWA not found for {dset}")
        except Exception as e:
            print(f"Error creating SWA plot for {dset}: {e}")
            plt.close()
        # ---------- print metrics ----------
        print(
            f'{dset} TEST: loss={test_metrics.get("loss"):.4f}, swa={test_metrics.get("swa"):.3f}'
        )
