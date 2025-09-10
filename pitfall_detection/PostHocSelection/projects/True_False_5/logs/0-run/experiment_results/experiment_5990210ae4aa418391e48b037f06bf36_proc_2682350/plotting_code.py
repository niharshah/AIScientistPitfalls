import matplotlib.pyplot as plt
import numpy as np
import os

# -------- paths --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    for dset_name, rec in experiment_data.items():
        # pull arrays safely
        train_loss = np.asarray(rec["losses"].get("train", []), dtype=float)
        val_loss = np.asarray(rec["losses"].get("val", []), dtype=float)
        train_swa = np.asarray(rec["metrics"].get("train_swa", []), dtype=float)
        val_swa = np.asarray(rec["metrics"].get("val_swa", []), dtype=float)
        preds = np.asarray(rec.get("predictions", []))
        gts = np.asarray(rec.get("ground_truth", []))
        epochs = np.arange(1, len(train_loss) + 1)

        # ---- 1: loss curves ----
        try:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset_name}: Train vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset_name}: {e}")
            plt.close()

        # ---- 2: SWA curves ----
        try:
            plt.figure()
            plt.plot(epochs, train_swa, label="Train SWA")
            plt.plot(epochs, val_swa, label="Validation SWA")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{dset_name}: Train vs Validation SWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_swa_curves.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating SWA plot for {dset_name}: {e}")
            plt.close()

        # ---- 3: test accuracy bar ----
        try:
            test_acc = float(np.mean(preds == gts)) if len(preds) else np.nan
            plt.figure()
            plt.bar(["Accuracy"], [test_acc])
            plt.ylim(0, 1)
            plt.title(f"{dset_name}: Test Accuracy")
            fname = os.path.join(working_dir, f"{dset_name}_test_accuracy.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {dset_name}: {e}")
            plt.close()

        # ---- print summary ----
        best_val_swa = float(np.max(val_swa)) if len(val_swa) else np.nan
        test_swa = (
            float(np.mean(train_swa[-1:])) if len(train_swa) else np.nan
        )  # placeholder
        print(f"{dset_name}: best_val_SWA={best_val_swa:.4f}, test_acc={test_acc:.4f}")
