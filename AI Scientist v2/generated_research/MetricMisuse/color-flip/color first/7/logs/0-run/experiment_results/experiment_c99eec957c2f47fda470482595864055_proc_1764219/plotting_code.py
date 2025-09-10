import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

saved_files = []

# -------------------- iterate & plot --------------------
for model_name, ds_dict in experiment_data.items():
    for ds_name, rec in ds_dict.items():
        epochs = np.array(rec.get("epochs", []))
        if epochs.size == 0:
            continue

        # ---- helper to unpack metric into np.array ----
        def metric_series(split, key):
            lst = rec["metrics"].get(split, [])
            return np.array([m.get(key, np.nan) for m in lst]) if lst else np.array([])

        # ========= 1. Loss curve =========
        try:
            tr_loss = np.array(rec["losses"].get("train", []))
            val_loss = np.array(rec["losses"].get("val", []))
            if tr_loss.size and val_loss.size:
                plt.figure()
                plt.plot(epochs, tr_loss, label="Train")
                plt.plot(epochs, val_loss, label="Validation")
                plt.xlabel("Epoch"), plt.ylabel("Loss")
                plt.title(f"{ds_name}: Train vs Validation Loss")
                plt.legend()
                fname = f"{ds_name}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                saved_files.append(fname)
            plt.close()
        except Exception as e:
            print(f"Error plotting loss curve for {ds_name}: {e}")
            plt.close()

        # ========= 2-4. Metric curves =========
        for metric in ["cwa", "swa", "cpx"]:
            try:
                tr_m = metric_series("train", metric)
                val_m = metric_series("val", metric)
                if tr_m.size and val_m.size:
                    plt.figure()
                    plt.plot(epochs, tr_m, label="Train")
                    plt.plot(epochs, val_m, label="Validation")
                    plt.xlabel("Epoch")
                    plt.ylabel(metric.upper())
                    plt.title(f"{ds_name}: Train vs Validation {metric.upper()}")
                    plt.legend()
                    fname = f"{ds_name}_{metric}_curve.png"
                    plt.savefig(os.path.join(working_dir, fname))
                    saved_files.append(fname)
                plt.close()
            except Exception as e:
                print(f"Error plotting {metric} for {ds_name}: {e}")
                plt.close()

        # ========= 5. Confusion matrix (val) =========
        try:
            preds = np.array(rec.get("predictions", []))
            gts = np.array(rec.get("ground_truth", []))
            if preds.size and gts.size:
                num_classes = int(max(preds.max(), gts.max()) + 1)
                cm = np.zeros((num_classes, num_classes), dtype=int)
                for p, t in zip(preds, gts):
                    cm[t, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xlabel("Predicted label"), plt.ylabel("True label")
                plt.title(
                    f"{ds_name}: Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
                )
                fname = f"{ds_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                saved_files.append(fname)
                plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix for {ds_name}: {e}")
            plt.close()

print("Saved figures:", saved_files)
