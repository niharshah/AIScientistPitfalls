import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_val_swa_all = {}
for dname, rec in experiment_data.items():
    # -------- extract with robust fallback --------
    train_loss = np.asarray(rec["losses"].get("train", []), dtype=float)
    val_loss = np.asarray(rec["losses"].get("val", []), dtype=float)
    train_swa = np.asarray(rec["metrics"].get("train_swa", []), dtype=float)
    val_swa = np.asarray(rec["metrics"].get("val_swa", []), dtype=float)
    preds = np.asarray(rec.get("predictions", []))
    gts = np.asarray(rec.get("ground_truth", []))
    epochs = np.arange(1, len(train_loss) + 1)

    # cache best val swa for later comparison
    if len(val_swa):
        best_val_swa_all[dname] = float(np.nanmax(val_swa))

    # -------- plot 1: loss curves --------
    try:
        plt.figure()
        if len(train_loss):
            plt.plot(epochs, train_loss, label="Train")
        if len(val_loss):
            plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname}: Train vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # -------- plot 2: SWA curves --------
    try:
        plt.figure()
        if len(train_swa):
            plt.plot(epochs, train_swa, label="Train SWA")
        if len(val_swa):
            plt.plot(epochs, val_swa, label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{dname}: Train vs Validation SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_swa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {dname}: {e}")
        plt.close()

    # -------- plot 3: test accuracy --------
    try:
        acc = float(np.mean(preds == gts)) if len(preds) else np.nan
        plt.figure()
        plt.bar(["Accuracy"], [acc])
        plt.ylim(0, 1)
        plt.title(f"{dname}: Test Accuracy")
        plt.savefig(os.path.join(working_dir, f"{dname}_test_accuracy.png"))
        plt.close()
        print(f"{dname}  Test Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
        plt.close()

# -------- comparison plot (best val SWA across datasets) --------
if len(best_val_swa_all) > 1:
    try:
        plt.figure()
        names = list(best_val_swa_all.keys())
        vals = [best_val_swa_all[n] for n in names]
        plt.bar(names, vals)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.title("Best Validation SWA Comparison Across Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "comparison_best_val_swa.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()
