import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- SETUP -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

for ds_name, ds in exp.items():
    # ----------- QUICK METRIC PRINT -----------
    tst_swa = ds["metrics"].get("test", None)
    if tst_swa is not None:
        print(f"{ds_name} – final test SWA: {tst_swa:.4f}")

    epochs = np.arange(len(ds["losses"]["train"]))
    step = max(1, len(epochs) // 50)  # subsampling interval

    # ----------- 1) LOSS CURVES -----------
    try:
        plt.figure()
        plt.plot(epochs, ds["losses"]["train"], "--", label="train")
        plt.plot(epochs, ds["losses"]["val"], "-", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Loss Curves\nTrain (dashed) vs Validation (solid)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # ----------- 2) SWA CURVES -----------
    try:
        plt.figure()
        plt.plot(
            epochs[::step],
            np.array(ds["metrics"]["train"])[::step],
            "--",
            label="train",
        )
        plt.plot(
            epochs[::step], np.array(ds["metrics"]["val"])[::step], "-", label="val"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Acc.")
        plt.title(f"{ds_name} SWA Curves\nTrain (dashed) vs Validation (solid)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_swa_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {ds_name}: {e}")
        plt.close()

    # ----------- 3) TRAIN-VS-VAL SWA SCATTER -----------
    try:
        plt.figure()
        plt.scatter(
            ds["metrics"]["train"], ds["metrics"]["val"], c=epochs, cmap="viridis", s=25
        )
        plt.colorbar(label="Epoch")
        plt.xlabel("Train SWA")
        plt.ylabel("Val SWA")
        plt.title(f"{ds_name} Train vs Validation SWA")
        fname = os.path.join(working_dir, f"{ds_name.lower()}_train_vs_val_swa.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot for {ds_name}: {e}")
        plt.close()

    # ----------- 4) FINAL TEST SWA BAR -----------
    try:
        plt.figure()
        plt.bar([ds_name], [tst_swa], color="skyblue")
        plt.ylabel("Shape-Weighted Acc.")
        plt.ylim(0, 1)
        plt.title(f"{ds_name} Final Test SWA")
        fname = os.path.join(working_dir, f"{ds_name.lower()}_test_swa_bar.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart for {ds_name}: {e}")
        plt.close()

    # ----------- 5) PREDICTION HISTOGRAM -----------
    try:
        preds = ds.get("predictions", [])
        if preds:
            uniq, cnts = np.unique(preds, return_counts=True)
            plt.figure()
            plt.bar(uniq, cnts, color="tab:orange")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Count")
            plt.title(f"{ds_name} Prediction Distribution – Test Set")
            fname = os.path.join(working_dir, f"{ds_name.lower()}_pred_hist.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error creating histogram for {ds_name}: {e}")
        plt.close()
