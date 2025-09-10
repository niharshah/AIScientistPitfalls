import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- load data ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to unpack (epoch, val) pairs -----------------
def unpack(store, *path):
    cur = store
    for p in path:
        cur = cur[p]
    if not cur:
        return np.array([]), np.array([])
    ep, vals = zip(*cur)
    return np.array(ep), np.array(vals)


plot_count, max_plots = 0, 5
for run_name, run_store in experiment_data.items():
    dataset_name = run_name  # only key we have
    # 1) loss curve -----------------------------------
    if plot_count < max_plots:
        try:
            ep_tr, tr_loss = unpack(run_store, "losses", "train")
            ep_va, va_loss = unpack(run_store, "losses", "val")
            plt.figure()
            plt.plot(ep_tr, tr_loss, label="Train")
            plt.plot(ep_va, va_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"Loss Curves ({dataset_name})")
            plt.legend()
            fname = f"{dataset_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting loss curves: {e}")
            plt.close()
        plot_count += 1
    # 2) weighted metric curves ------------------------
    if plot_count < max_plots:
        try:
            ep, swa, cwa, comp = [], [], [], []
            for tup in run_store["metrics"]["train"]:
                ep.append(tup[0])
                swa.append(tup[1])
                cwa.append(tup[2])
                comp.append(tup[3])
            plt.figure()
            plt.plot(ep, swa, label="SWA")
            plt.plot(ep, cwa, label="CWA")
            plt.plot(ep, comp, label="CompWA")
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title(f"SWA / CWA / CompWA ({dataset_name})")
            plt.legend()
            fname = f"{dataset_name}_weighted_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting weighted metrics: {e}")
            plt.close()
        plot_count += 1
    # 3) simple accuracy curve -------------------------
    if plot_count < max_plots:
        try:
            acc_ep, acc_vals = [], []
            for ep_idx, preds in run_store["predictions"]:
                gts = dict(run_store["ground_truth"])[ep_idx]
                preds = np.array(preds)
                gts = np.array(gts)
                acc = (preds == gts).mean() if len(preds) else 0.0
                acc_ep.append(ep_idx)
                acc_vals.append(acc)
            plt.figure()
            plt.plot(acc_ep, acc_vals, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Validation Accuracy ({dataset_name})")
            fname = f"{dataset_name}_accuracy_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting accuracy curve: {e}")
            plt.close()
        plot_count += 1
    # 4) confusion matrix final epoch ------------------
    if plot_count < max_plots:
        try:
            last_ep = max(ep for ep, _ in run_store["predictions"])
            preds = dict(run_store["predictions"])[last_ep]
            gts = dict(run_store["ground_truth"])[last_ep]
            cm = np.zeros((2, 2), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.title(f"Confusion Matrix Final Epoch ({dataset_name})")
            fname = f"{dataset_name}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            plt.close()
        plot_count += 1
