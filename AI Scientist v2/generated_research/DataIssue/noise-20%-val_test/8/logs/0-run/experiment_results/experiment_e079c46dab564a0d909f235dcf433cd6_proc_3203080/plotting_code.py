import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- Iterate over datasets ----------
for exp_name, datasets in experiment_data.items():  # e.g. 'max_features_tuning'
    for dset_name, dset_dict in datasets.items():  # e.g. 'SPR_BENCH'
        # Short aliases
        hp = dset_dict.get("hyperparams", [])
        tr_acc = dset_dict.get("metrics", {}).get("train", [])
        va_acc = dset_dict.get("metrics", {}).get("val", [])
        tr_loss = dset_dict.get("losses", {}).get("train", [])
        va_loss = dset_dict.get("losses", {}).get("val", [])
        preds = np.array(dset_dict.get("predictions", []))
        gts = np.array(dset_dict.get("ground_truth", []))
        test_acc = dset_dict.get("test_accuracy", None)
        sefa_val = dset_dict.get("sefa", None)

        # 1) Accuracy curve
        try:
            plt.figure()
            plt.plot(hp, tr_acc, marker="o", label="Train")
            plt.plot(hp, va_acc, marker="s", label="Validation")
            plt.xlabel("max_features setting")
            plt.ylabel("Accuracy")
            plt.title(f"{dset_name}: Train vs Val Accuracy")
            plt.legend()
            fname = f"{dset_name}_accuracy_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy curve for {dset_name}: {e}")
            plt.close()

        # 2) Loss curve
        try:
            plt.figure()
            plt.plot(hp, tr_loss, marker="o", label="Train")
            plt.plot(hp, va_loss, marker="s", label="Validation")
            plt.xlabel("max_features setting")
            plt.ylabel("Loss (1-Acc)")
            plt.title(f"{dset_name}: Train vs Val Loss")
            plt.legend()
            fname = f"{dset_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dset_name}: {e}")
            plt.close()

        # 3) Bar chart of Test Accuracy & SEFA
        try:
            metrics_names = []
            metrics_vals = []
            if test_acc is not None:
                metrics_names.append("Test Accuracy")
                metrics_vals.append(test_acc)
            if sefa_val is not None:
                metrics_names.append("SEFA")
                metrics_vals.append(sefa_val)
            if metrics_vals:
                plt.figure()
                plt.bar(metrics_names, metrics_vals, color=["skyblue", "lightgreen"])
                plt.ylim(0, 1)
                plt.title(f"{dset_name}: Final Test Metrics")
                fname = f"{dset_name}_test_metrics.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating test metrics bar for {dset_name}: {e}")
            plt.close()

        # 4) Confusion matrix (single plot)
        try:
            if preds.size and gts.size:
                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(gts, preds)
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{dset_name}: Confusion Matrix")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                        )
                fname = f"{dset_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dset_name}: {e}")
            plt.close()

        # Print final evaluation metrics
        if test_acc is not None:
            print(f"{dset_name} - Test Accuracy: {test_acc:.4f}")
        if sefa_val is not None:
            print(f"{dset_name} - SEFA: {sefa_val:.4f}")
