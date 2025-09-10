import matplotlib.pyplot as plt
import numpy as np
import os

# --- setup -------------------------------------------------------------------
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
    for dname, record in experiment_data.items():  # e.g. 'SPR_BENCH'
        # ------------------------------------------------- 1) loss curves
        try:
            train_l = record["losses"]["train"]
            val_l = record["losses"]["val"]
            if train_l and val_l:
                epochs = range(1, len(train_l) + 1)
                plt.figure(figsize=(6, 4))
                plt.plot(epochs, train_l, "b-o", label="Train")
                plt.plot(epochs, val_l, "r-o", label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{dname}: Train vs Val Loss")
                plt.legend()
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
                plt.close()
        except Exception as e:
            print(f"Error plotting loss curves for {dname}: {e}")
            plt.close()

        # ------------------------------------------------- 2) val metric (SWA) curves
        try:
            val_metrics = record["metrics"]["val"]
            val_swa = [m.get("swa") for m in val_metrics if "swa" in m]
            if val_swa:
                epochs = range(1, len(val_swa) + 1)
                plt.figure(figsize=(6, 4))
                plt.plot(epochs, val_swa, "g-s")
                plt.ylim(0, 1)
                plt.xlabel("Epoch")
                plt.ylabel("SWA")
                plt.title(f"{dname}: Validation Shape-Weighted Accuracy")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{dname}_val_swa.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
                plt.close()
        except Exception as e:
            print(f"Error plotting SWA for {dname}: {e}")
            plt.close()

        # ------------------------------------------------- 3) final test SWA bar
        try:
            test_swa = record["metrics"]["test"].get("swa")
            if test_swa is not None:
                plt.figure(figsize=(4, 3))
                plt.bar(["Test SWA"], [test_swa], color="coral")
                plt.ylim(0, 1)
                plt.ylabel("SWA")
                plt.title(f"{dname}: Final Test SWA")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{dname}_test_swa_bar.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
                plt.close()
        except Exception as e:
            print(f"Error plotting test SWA for {dname}: {e}")
            plt.close()

        # ------------------------------------------------- 4) confusion matrix
        try:
            preds = record.get("predictions", [])
            trues = record.get("ground_truth", [])
            if preds and trues:
                n_lab = max(max(preds), max(trues)) + 1
                conf = np.zeros((n_lab, n_lab), dtype=int)
                for t, p in zip(trues, preds):
                    conf[t, p] += 1
                # If too large, slice to 5x5
                if n_lab > 5:
                    conf = conf[:5, :5]
                    n_lab = 5
                plt.figure(figsize=(4, 4))
                plt.imshow(conf, cmap="Blues")
                plt.colorbar()
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(
                    f"{dname}: Confusion Matrix (truncated)"
                    if n_lab < max(preds) + 1
                    else f"{dname}: Confusion Matrix"
                )
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
                plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix for {dname}: {e}")
            plt.close()
