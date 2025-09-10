import matplotlib.pyplot as plt
import numpy as np
import os

# Re-establish working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset, data in experiment_data.items():
    # Helper to fetch list safely
    def safe_get(path, default=None):
        cur = data
        for k in path:
            cur = cur.get(k, None) if isinstance(cur, dict) else None
            if cur is None:
                return default
        return cur

    # 1) Supervised loss curves ------------------------------------------------
    try:
        tr_loss = safe_get(["losses", "supervised", "train"])
        val_loss = safe_get(["losses", "supervised", "val"])
        if tr_loss and val_loss:
            plt.figure()
            epochs = np.arange(1, len(tr_loss) + 1)
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset} Supervised Loss\nTrain vs Validation")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_supervised_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating supervised loss plot for {dset}: {e}")
        plt.close()

    # 2) Contrastive loss ------------------------------------------------------
    try:
        contrast = safe_get(["losses", "contrastive"])
        if contrast:
            plt.figure()
            plt.plot(np.arange(1, len(contrast) + 1), contrast, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            plt.title(f"{dset} Contrastive Pre-training Loss")
            fname = os.path.join(working_dir, f"{dset}_contrastive_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating contrastive loss plot for {dset}: {e}")
        plt.close()

    # Metric extractor ---------------------------------------------------------
    metrics_train = safe_get(["metrics", "train"], [])
    metrics_val = safe_get(["metrics", "val"], [])

    def metric_series(key, source):
        return [m.get(key, np.nan) for m in source] if source else []

    # 3) HWA curves ------------------------------------------------------------
    try:
        tr_hwa, val_hwa = metric_series("hwa", metrics_train), metric_series(
            "hwa", metrics_val
        )
        if tr_hwa and val_hwa:
            plt.figure()
            epochs = np.arange(1, len(tr_hwa) + 1)
            plt.plot(epochs, tr_hwa, label="Train")
            plt.plot(epochs, val_hwa, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Harmonic Weighted Acc")
            plt.title(f"{dset} HWA\nTrain vs Validation")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_HWA.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating HWA plot for {dset}: {e}")
        plt.close()

    # 4) SWA curves ------------------------------------------------------------
    try:
        tr_swa, val_swa = metric_series("swa", metrics_train), metric_series(
            "swa", metrics_val
        )
        if tr_swa and val_swa:
            plt.figure()
            epochs = np.arange(1, len(tr_swa) + 1)
            plt.plot(epochs, tr_swa, label="Train")
            plt.plot(epochs, val_swa, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Shape Weighted Acc")
            plt.title(f"{dset} SWA\nTrain vs Validation")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_SWA.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {dset}: {e}")
        plt.close()

    # 5) CWA curves ------------------------------------------------------------
    try:
        tr_cwa, val_cwa = metric_series("cwa", metrics_train), metric_series(
            "cwa", metrics_val
        )
        if tr_cwa and val_cwa:
            plt.figure()
            epochs = np.arange(1, len(tr_cwa) + 1)
            plt.plot(epochs, tr_cwa, label="Train")
            plt.plot(epochs, val_cwa, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Color Weighted Acc")
            plt.title(f"{dset} CWA\nTrain vs Validation")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_CWA.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating CWA plot for {dset}: {e}")
        plt.close()

    # Print final validation metrics ------------------------------------------
    try:
        if metrics_val:
            last = metrics_val[-1]
            print(
                f"{dset} final validation metrics: SWA={last.get('swa'):.3f}, "
                f"CWA={last.get('cwa'):.3f}, HWA={last.get('hwa'):.3f}"
            )
    except Exception as e:
        print(f"Error printing final metrics for {dset}: {e}")
