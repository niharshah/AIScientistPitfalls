import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate over datasets ----------
for dset_name, dset in experiment_data.items():
    print(f"\n=== Processing {dset_name} ===")

    # 1) loss curves ----------------------------------------------------------
    try:
        tr_loss = dset["losses"].get("train", [])
        val_loss = dset["losses"].get("val", [])
        if tr_loss and val_loss:
            plt.figure()
            epochs = np.arange(1, len(tr_loss) + 1)
            plt.plot(epochs, tr_loss, label="train")
            plt.plot(epochs, val_loss, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_name} – Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {dset_name}: {e}")
        plt.close()

    # 2) validation metric curves --------------------------------------------
    try:
        val_metric = dset["metrics"].get("val", [])
        if val_metric:
            plt.figure()
            epochs = np.arange(1, len(val_metric) + 1)
            plt.plot(epochs, val_metric, marker="o", color="tab:green")
            plt.xlabel("Epoch")
            plt.ylabel("CompWA")
            plt.title(f"{dset_name} – Validation CompWA over Epochs")
            fname = os.path.join(working_dir, f"{dset_name}_val_metric_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting metric curve for {dset_name}: {e}")
        plt.close()

    # 3) confusion matrix -----------------------------------------------------
    try:
        preds = np.array(dset.get("predictions", []))
        gts = np.array(dset.get("ground_truth", []))
        if preds.size and gts.size:
            num_cls = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            acc = (preds == gts).mean()
            print(f"{dset_name} – overall accuracy: {acc:.4f}")

            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.ylabel("Ground Truth")
            plt.xlabel("Predicted")
            plt.title(f"{dset_name} – Confusion Matrix\n(rows=GT, cols=Pred)")
            fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset_name}: {e}")
        plt.close()
