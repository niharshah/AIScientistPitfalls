import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_data = {}

# -------------------- iterate and plot --------------------
for exp_name, datasets_dict in exp_data.items():
    for dset_name, rec in datasets_dict.items():
        epochs = rec.get("epochs", [])
        train_loss = rec.get("losses", {}).get("train", [])
        val_loss = rec.get("losses", {}).get("val", [])
        train_f1 = rec.get("metrics", {}).get("train", [])
        val_f1 = rec.get("metrics", {}).get("val", [])
        test_f1 = rec.get("test_macro_f1", None)
        best_val_f1 = max(val_f1) if val_f1 else None
        test_loss = rec.get("test_loss", None)

        # 1) Loss curves
        try:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_name} – Training vs. Validation Loss")
            plt.legend()
            fname = f"{exp_name}_{dset_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset_name}: {e}")
            plt.close()

        # 2) F1 curves
        try:
            plt.figure()
            plt.plot(epochs, train_f1, label="Train Macro-F1")
            plt.plot(epochs, val_f1, label="Val Macro-F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dset_name} – Training vs. Validation Macro-F1")
            plt.legend()
            fname = f"{exp_name}_{dset_name}_f1_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating F1 plot for {dset_name}: {e}")
            plt.close()

        # 3) Test vs Best-Val bar
        try:
            if test_f1 is not None and best_val_f1 is not None:
                plt.figure()
                plt.bar(
                    ["Best Val", "Test"],
                    [best_val_f1, test_f1],
                    color=["skyblue", "salmon"],
                )
                plt.ylabel("Macro-F1")
                plt.title(f"{dset_name} – Best Validation vs. Test Macro-F1")
                fname = f"{exp_name}_{dset_name}_val_vs_test_f1.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating bar plot for {dset_name}: {e}")
            plt.close()

        # ------------- print metrics -------------
        print(
            f"{exp_name} | {dset_name} -> Test Loss: {test_loss:.4f} | Test Macro-F1: {test_f1:.4f}"
        )
