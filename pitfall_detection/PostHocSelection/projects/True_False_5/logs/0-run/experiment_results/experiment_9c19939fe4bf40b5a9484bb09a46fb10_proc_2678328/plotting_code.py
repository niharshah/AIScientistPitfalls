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
    experiment_data = {}

val_swa_collect = {}  # for cross-dataset comparison
if experiment_data:
    for dname, rec in experiment_data.items():
        # -------- extract arrays ----------
        train_loss = np.asarray(rec["losses"].get("train", []))
        val_loss = np.asarray(rec["losses"].get("val", []))
        train_swa = np.asarray(rec["metrics"].get("train_SWA", []))
        val_swa = np.asarray(rec["metrics"].get("val_SWA", []))
        preds = np.asarray(rec.get("predictions", []))
        gts = np.asarray(rec.get("ground_truth", []))
        epochs = np.arange(1, len(train_loss) + 1)

        val_swa_collect[dname] = val_swa

        # -------- plot 1: loss curves --------
        try:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train")
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
            plt.plot(epochs, train_swa, label="Train SWA")
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

        # -------- plot 3: test accuracy bar --------
        try:
            test_acc = float(np.mean(preds == gts)) if len(preds) else np.nan
            plt.figure()
            plt.bar(["Accuracy"], [test_acc])
            plt.ylim(0, 1)
            plt.title(f"{dname}: Test Accuracy")
            plt.savefig(os.path.join(working_dir, f"{dname}_test_accuracy.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {dname}: {e}")
            plt.close()

        # -------- plot 4: confusion matrix --------
        try:
            if len(preds) and len(gts):
                num_cls = int(max(np.max(preds), np.max(gts))) + 1
                cm = np.zeros((num_cls, num_cls), dtype=int)
                for t, p in zip(gts, preds):
                    cm[int(t), int(p)] += 1
                plt.figure(figsize=(4, 4))
                plt.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{dname}: Confusion Matrix")
                plt.colorbar()
                plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dname}: {e}")
            plt.close()

        # ----- print evaluation metrics -----
        if len(preds):
            print(f"{dname} - Test Accuracy: {np.mean(preds==gts):.4f}")

# -------- comparison plot across datasets (val SWA) --------
try:
    if len(val_swa_collect) > 1:
        plt.figure()
        for dname, vswa in val_swa_collect.items():
            plt.plot(np.arange(1, len(vswa) + 1), vswa, label=dname)
        plt.xlabel("Epoch")
        plt.ylabel("Validation SWA")
        plt.title("Validation SWA Comparison Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "datasets_val_swa_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error creating cross-dataset SWA plot: {e}")
    plt.close()
