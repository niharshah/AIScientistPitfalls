import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------- load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    algo = next(iter(experiment_data))  # 'no_sequence_packing'
    dset = next(iter(experiment_data[algo]))  # 'SPR_BENCH'
    record = experiment_data[algo][dset]

    # ---------- Helper to unpack (epoch, v) tuples -------------------------
    def tup2arr(tups):
        ep, val = zip(*tups) if tups else ([], [])
        return np.array(ep), np.array(val)

    # ------------------------- FIGURE 1: Loss curves -----------------------
    try:
        tr_ep, tr_loss = tup2arr(record["losses"]["train"])
        va_ep, va_loss = tup2arr(record["losses"]["val"])

        plt.figure()
        plt.plot(tr_ep, tr_loss, label="Train")
        plt.plot(va_ep, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset} Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset.lower()}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------------- FIGURE 2: Validation metric trajectories ---------------
    try:
        if record["metrics"]["val"]:
            vals = np.array(record["metrics"]["val"])
            ep = vals[:, 0]
            labels = ["CWA", "SWA", "HCSA", "SNWA"]
            for i, lab in enumerate(labels, start=1):
                plt.plot(ep, vals[:, i], label=lab)
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dset} Validation Metrics")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset.lower()}_val_metrics.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # -------------- Helper to create confusion matrix plots ----------------
    def plot_cm(split):
        preds = np.array(record["predictions"][split])
        gts = np.array(record["ground_truth"][split])
        if preds.size == 0 or gts.size == 0:
            return
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{dset} Confusion Matrix ({split})")
        fname = os.path.join(
            working_dir, f"{dset.lower()}_confusion_matrix_{split}.png"
        )
        plt.savefig(fname)
        plt.close()

    # --------------------- FIGURE 3 & 4: Confusion matrices ---------------
    for split in ["dev", "test"]:
        try:
            plot_cm(split)
        except Exception as e:
            print(f"Error creating confusion matrix ({split}): {e}")
            plt.close()
