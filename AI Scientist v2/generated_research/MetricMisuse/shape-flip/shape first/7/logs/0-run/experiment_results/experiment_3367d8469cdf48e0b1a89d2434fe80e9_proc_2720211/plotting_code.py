import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
test_swa_by_ds = {}

for ds_name, ds_blob in experiment_data.items():
    # ----------------- gather data -------------------------------
    tr_loss = ds_blob.get("losses", {}).get("train", [])
    vl_loss = ds_blob.get("losses", {}).get("val", [])
    tr_met = ds_blob.get("metrics", {}).get("train", [])
    vl_met = ds_blob.get("metrics", {}).get("val", [])

    epochs = [e for e, _ in tr_loss] if tr_loss else []
    tr_loss_vals = [v for _, v in tr_loss]
    vl_loss_vals = [v for _, v in vl_loss]
    tr_met_vals = [v for _, v in tr_met]
    vl_met_vals = [v for _, v in vl_met]

    gtruth = ds_blob.get("ground_truth", [])
    preds = ds_blob.get("predictions", [])

    swa_test = (
        sum(int(gt == pr) for gt, pr in zip(gtruth, preds)) / len(gtruth)
        if gtruth
        else np.nan
    )
    test_swa_by_ds[ds_name] = swa_test

    # ----------------- figure 1 : loss curves ---------------------
    try:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, tr_loss_vals, label="train")
        plt.title(f"{ds_name} – Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("CE Loss")
        plt.legend(fontsize="small")

        plt.subplot(2, 1, 2)
        plt.plot(epochs, vl_loss_vals, label="val")
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("CE Loss")
        plt.legend(fontsize="small")

        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {ds_name}: {e}")
        plt.close()

    # ---------------- figure 2 : metric curves --------------------
    try:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, tr_met_vals, label="train")
        plt.title(f"{ds_name} – Training SWA")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend(fontsize="small")

        plt.subplot(2, 1, 2)
        plt.plot(epochs, vl_met_vals, label="val")
        plt.title("Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend(fontsize="small")

        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting SWA curves for {ds_name}: {e}")
        plt.close()

    # ------------- figure 3 : prediction histogram ----------------
    try:
        if gtruth and preds:
            plt.figure(figsize=(6, 4))
            bins = np.arange(-0.5, max(max(gtruth), max(preds)) + 1.5, 1)
            plt.hist(gtruth, bins=bins, alpha=0.6, label="Ground Truth")
            plt.hist(preds, bins=bins, alpha=0.6, label="Predictions")
            plt.title(
                f"{ds_name} – Test Set Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.legend(fontsize="small")
            fname = os.path.join(working_dir, f"{ds_name.lower()}_test_histogram.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting histogram for {ds_name}: {e}")
        plt.close()

# ---------------- comparison plot across datasets -----------------
try:
    if len(test_swa_by_ds) > 1:
        plt.figure(figsize=(6, 4))
        names = list(test_swa_by_ds.keys())
        accs = [test_swa_by_ds[k] for k in names]
        plt.bar(names, accs)
        plt.title("Test SWA Across Datasets")
        plt.xlabel("Dataset")
        plt.ylabel("SWA")
        for i, a in enumerate(accs):
            plt.text(i, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        fname = os.path.join(working_dir, "datasets_test_swa_comparison.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error plotting dataset comparison: {e}")
    plt.close()

# --------------------- print metrics ------------------------------
print("Final Test SWA:")
for ds, swa in test_swa_by_ds.items():
    print(f"  {ds}: {swa:.4f}")
