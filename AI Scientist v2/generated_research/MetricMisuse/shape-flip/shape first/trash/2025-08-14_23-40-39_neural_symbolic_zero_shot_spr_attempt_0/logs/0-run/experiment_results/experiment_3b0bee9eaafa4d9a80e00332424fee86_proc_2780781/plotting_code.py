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


# ---------- helper ----------
def downsample(arr, max_pts=200):
    if len(arr) <= max_pts:
        return np.arange(len(arr)), arr
    idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
    return idx, np.array(arr)[idx]


datasets = list(experiment_data.keys())
best_swa_all = {}
for ds_name, ds_dict in experiment_data.items():
    metrics = ds_dict.get("metrics", {})
    ep = np.arange(1, len(metrics.get("train_loss", [])) + 1)

    # 1. loss curves -------------------------------------------------
    try:
        if ep.size:
            ds_idx, t_loss = downsample(metrics.get("train_loss", []))
            _, v_loss = downsample(metrics.get("val_loss", []))
            plt.figure()
            plt.plot(ep[ds_idx], t_loss, label="Train Loss")
            plt.plot(ep[ds_idx], v_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name}: Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2. validation SWA curve ---------------------------------------
    try:
        if "val_swa" in metrics:
            idx, v_swa = downsample(metrics["val_swa"])
            plt.figure()
            plt.plot(ep[idx], v_swa, label="Val SWA")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{ds_name}: Validation SWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_val_swa.png")
            plt.savefig(fname)
            best_swa_all[ds_name] = max(metrics["val_swa"])
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {ds_name}: {e}")
        plt.close()

    # 3. confusion matrix on test -----------------------------------
    try:
        preds = np.array(ds_dict.get("predictions", {}).get("test", []))
        gts = np.array(ds_dict.get("ground_truth", {}).get("test", []))
        if preds.size and gts.size:
            classes = np.unique(np.concatenate([gts, preds]))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"{ds_name}: Test Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(classes)
            plt.yticks(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{ds_name}_test_confusion_matrix.png")
            plt.savefig(fname)
            # simple accuracy
            acc = np.trace(cm) / cm.sum() if cm.sum() else 0.0
            print(
                f"{ds_name} | Best Val SWA: {best_swa_all.get(ds_name, np.nan):.4f} | Test Acc: {acc:.4f}"
            )
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

# 4. cross-dataset comparison plot ---------------------------------
try:
    if len(best_swa_all) > 1:
        plt.figure()
        names = list(best_swa_all.keys())
        scores = [best_swa_all[n] for n in names]
        plt.bar(names, scores)
        plt.ylabel("Best Validation SWA")
        plt.title("Dataset Comparison: Best Validation SWA")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = os.path.join(working_dir, "datasets_best_val_swa_comparison.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset comparison plot: {e}")
    plt.close()
