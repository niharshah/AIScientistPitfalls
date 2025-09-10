import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def _safe(arr, k, default=np.nan):
    return np.array(arr.get(k, []), dtype=float)


# ---------- Iterate & plot ----------
for dset, dct in experiment_data.items():
    for exp_name, ed in dct.items():
        epochs = np.array(ed.get("epochs", []))
        losses = ed.get("losses", {})
        mets = ed.get("metrics", {})
        gt = np.array(ed.get("ground_truth", []))
        pred = np.array(ed.get("predictions", []))

        # 1) Loss curves ----------------------------------------------------
        try:
            plt.figure()
            plt.plot(epochs, _safe(losses, "train"), label="Train")
            plt.plot(epochs, _safe(losses, "val"), label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset}-{exp_name} Loss Curves")
            plt.legend()
            fname = f"{dset}_{exp_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error plotting loss curve for {dset}-{exp_name}: {e}")
            plt.close()

        # 2) HWA over epochs ----------------------------------------------
        try:
            hwa_tr = [m.get("HWA", np.nan) for m in mets.get("train", [])]
            hwa_va = [m.get("HWA", np.nan) for m in mets.get("val", [])]
            plt.figure()
            plt.plot(epochs, hwa_tr, label="Train")
            plt.plot(epochs, hwa_va, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("HWA")
            plt.title(f"{dset}-{exp_name} HWA Curves")
            plt.legend()
            fname = f"{dset}_{exp_name}_hwa_curve.png"
            plt.savefig(os.path.join(working_dir, fname), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error plotting HWA curve for {dset}-{exp_name}: {e}")
            plt.close()

        # 3) Scatter GT vs Pred -------------------------------------------
        try:
            if gt.size and pred.size:
                plt.figure()
                jitter = (np.random.rand(len(gt)) - 0.5) * 0.2
                plt.scatter(gt + jitter, pred + jitter, alpha=0.5)
                max_lab = int(max(gt.max(), pred.max()))
                plt.plot([0, max_lab], [0, max_lab], "k--", linewidth=1)
                plt.xlabel("Ground Truth")
                plt.ylabel("Prediction")
                plt.title(
                    f"{dset}-{exp_name}\nLeft: Ground Truth, Right: Generated Samples (Test)"
                )
                fname = f"{dset}_{exp_name}_scatter_gt_vs_pred.png"
                plt.savefig(os.path.join(working_dir, fname), dpi=150)
                plt.close()
        except Exception as e:
            print(f"Error plotting scatter for {dset}-{exp_name}: {e}")
            plt.close()

        # 4) Bar chart of class distribution & accuracy --------------------
        try:
            if gt.size and pred.size:
                classes = np.arange(int(max(gt.max(), pred.max()) + 1))
                gt_counts = np.bincount(gt, minlength=len(classes))
                pred_counts = np.bincount(pred, minlength=len(classes))
                width = 0.35
                plt.figure()
                plt.bar(classes - width / 2, gt_counts, width, label="Ground Truth")
                plt.bar(classes + width / 2, pred_counts, width, label="Predictions")
                acc = (gt == pred).mean() if len(gt) else np.nan
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title(f"{dset}-{exp_name} Class Distribution (Acc={acc:.3f})")
                plt.legend()
                fname = f"{dset}_{exp_name}_class_dist.png"
                plt.savefig(os.path.join(working_dir, fname), dpi=150)
                plt.close()
                print(f"{dset}-{exp_name} test accuracy: {acc:.3f}")
        except Exception as e:
            print(f"Error plotting class distribution for {dset}-{exp_name}: {e}")
            plt.close()
