import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------- iterate over datasets -----------
for dset_name, rec in experiment_data.items():
    losses_tr = rec.get("losses", {}).get("train", [])  # list of (ep, val)
    losses_val = rec.get("losses", {}).get("val", [])  # may be empty
    metrics_val = rec.get("metrics", {}).get("val", [])  # list of (ep,swa,cwa,dawa)
    preds = np.array(rec.get("predictions", []))
    gts = np.array(rec.get("ground_truth", []))

    # ---------- 1. training loss ----------
    try:
        if losses_tr:
            ep_tr, val_tr = zip(*losses_tr)
            plt.figure()
            plt.plot(ep_tr, val_tr, label="train loss")
            if losses_val:
                ep_v, val_v = zip(*losses_val)
                plt.plot(ep_v, val_v, linestyle="--", label="val loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset_name}: Training (and Validation) Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
        else:
            print(f"No loss data for {dset_name}")
    except Exception as e:
        print(f"Error creating loss plot for {dset_name}: {e}")
        plt.close()

    # ---------- 2. metric curves ----------
    try:
        if metrics_val:
            ep, swa, cwa, dawa = zip(*metrics_val)
            plt.figure()
            plt.plot(ep, swa, label="SWA")
            plt.plot(ep, cwa, label="CWA")
            plt.plot(ep, dawa, label="DAWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dset_name}: Validation Metrics Over Time")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_metric_curves.png")
            plt.savefig(fname)
            plt.close()
        else:
            print(f"No metric data for {dset_name}")
    except Exception as e:
        print(f"Error creating metric plot for {dset_name}: {e}")
        plt.close()

    # ---------- 3. confusion matrix --------
    try:
        if preds.size and gts.size and preds.shape == gts.shape:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, labels=sorted(set(gts)))
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(f"{dset_name}: Confusion Matrix (final epoch)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(ticks=range(cm.shape[0]))
            plt.yticks(ticks=range(cm.shape[0]))
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        else:
            print(f"No predictions to plot confusion matrix for {dset_name}")
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()

    # ---------- 4. console summary ---------
    if metrics_val:
        print(f"{dset_name}: Final-epoch DAWA = {metrics_val[-1][-1]:.4f}")
