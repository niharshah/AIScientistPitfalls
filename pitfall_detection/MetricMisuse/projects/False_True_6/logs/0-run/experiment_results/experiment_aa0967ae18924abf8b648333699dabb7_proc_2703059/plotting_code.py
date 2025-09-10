import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load experiment artefacts ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

# ---- plotting ----
if experiment_data:
    for ds_name, rec in experiment_data.items():
        # ------------- 1) learning curves -------------
        try:
            epochs = range(1, len(rec["losses"]["train"]) + 1)
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, rec["losses"]["train"], "b-o", label="Train Loss")
            plt.plot(epochs, rec["losses"]["val"], "r-o", label="Val Loss")
            plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.legend()
            plt.title(f"{ds_name}: Left: Train Loss, Right: Val Loss")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds_name}_learning_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating learning curve for {ds_name}: {e}")
            plt.close()

        # ------------- 2) validation SWA -------------
        try:
            swa_vals = [m.get("swa") for m in rec["metrics"]["val"] if "swa" in m]
            if swa_vals:
                epochs = range(1, len(swa_vals) + 1)
                plt.figure(figsize=(6, 4))
                plt.plot(epochs, swa_vals, "g-s")
                plt.ylim(0, 1)
                plt.xlabel("Epoch"), plt.ylabel("SWA")
                plt.title(f"{ds_name}: Validation Shape-Weighted Accuracy")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_val_swa.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating SWA plot for {ds_name}: {e}")
            plt.close()

        # ------------- 3) test SWA summary -------------
        try:
            test_swa = rec["metrics"]["test"].get("swa", None)
            if test_swa is not None:
                plt.figure(figsize=(4, 4))
                plt.bar([ds_name], [test_swa], color="purple")
                plt.ylim(0, 1)
                plt.title(f"{ds_name}: Final Test SWA")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_test_swa.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating test SWA bar for {ds_name}: {e}")
            plt.close()
