import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for exp_name, ds_dict in experiment_data.items():
    for ds_name, rec in ds_dict.items():
        epochs = np.array(rec.get("epochs", []))
        tr_loss = np.array(rec.get("losses", {}).get("train", []))
        val_loss = np.array(rec.get("losses", {}).get("val", []))
        tr_f1 = np.array(rec.get("metrics", {}).get("train", []))
        val_f1 = np.array(rec.get("metrics", {}).get("val", []))
        preds = np.array(rec.get("predictions", []))
        gts = np.array(rec.get("ground_truth", []))
        test_f1 = rec.get("test_macroF1", None)
        if test_f1 is not None:
            print(f"{exp_name}-{ds_name}  Test Macro-F1: {test_f1:.4f}")

        # 1) loss curves
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{exp_name} – {ds_name} Loss Curves")
            plt.legend()
            fn = f"{exp_name}_{ds_name}_loss_curves.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fn))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # 2) macro-F1 curves
        try:
            plt.figure()
            plt.plot(epochs, tr_f1, label="Train")
            plt.plot(epochs, val_f1, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{exp_name} – {ds_name} Macro-F1 Curves")
            plt.legend()
            fn = f"{exp_name}_{ds_name}_macroF1_curves.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fn))
            plt.close()
        except Exception as e:
            print(f"Error creating F1 plot: {e}")
            plt.close()

        # 3) confusion matrix
        try:
            if preds.size and gts.size:
                num_labels = int(max(gts.max(), preds.max()) + 1)
                cm = np.zeros((num_labels, num_labels), dtype=int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                plt.figure(figsize=(5, 5))
                im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{exp_name} – {ds_name} Confusion Matrix")
                plt.savefig(
                    os.path.join(
                        working_dir,
                        f"{exp_name}_{ds_name}_confusion_matrix.png".replace(" ", "_"),
                    )
                )
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

        # 4) class distribution comparison
        try:
            if preds.size and gts.size:
                num_labels = int(max(gts.max(), preds.max()) + 1)
                ind = np.arange(num_labels)
                width = 0.35
                gt_counts = np.bincount(gts, minlength=num_labels)
                pr_counts = np.bincount(preds, minlength=num_labels)
                plt.figure()
                plt.bar(ind - width / 2, gt_counts, width, label="Ground Truth")
                plt.bar(ind + width / 2, pr_counts, width, label="Predictions")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title(f"{exp_name} – {ds_name} Label Distribution")
                plt.legend()
                fn = f"{exp_name}_{ds_name}_label_distribution.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fn))
                plt.close()
        except Exception as e:
            print(f"Error creating distribution plot: {e}")
            plt.close()
