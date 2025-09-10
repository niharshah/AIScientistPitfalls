import matplotlib.pyplot as plt
import numpy as np
import os

# ----- paths -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load experiment data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----- container for comparison plot -----
val_swa_curves = {}

for ds_name, rec in experiment_data.items():
    # ---------------- extract arrays ----------------
    try:
        train_loss = np.asarray(rec["losses"]["train"])
        val_loss = np.asarray(rec["losses"]["val"])
    except Exception:
        train_loss = val_loss = np.array([])
    try:
        train_swa = np.asarray(rec["metrics"]["train_swa"])
        val_swa = np.asarray(rec["metrics"]["val_swa"])
    except Exception:
        train_swa = val_swa = np.array([])
    preds = np.asarray(rec.get("predictions", []))
    gts = np.asarray(rec.get("ground_truth", []))
    epochs = np.arange(1, len(train_loss) + 1)

    # store for comparison
    if len(val_swa):
        val_swa_curves[ds_name] = (epochs, val_swa)

    # --------- plot 1: loss curves ---------
    try:
        if len(train_loss) and len(val_loss):
            plt.figure()
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name}: Train vs Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # --------- plot 2: SWA curves ---------
    try:
        if len(train_swa) and len(val_swa):
            plt.figure()
            plt.plot(epochs, train_swa, label="Train SWA")
            plt.plot(epochs, val_swa, label="Validation SWA")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{ds_name}: Train vs Validation SWA")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_swa_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {ds_name}: {e}")
        plt.close()

    # --------- plot 3: Test metrics bar ---------
    try:
        if len(preds) and len(gts):
            acc = (preds == gts).mean()
            swa_test = rec["metrics"].get("test_swa", np.nan)
            plt.figure()
            plt.bar(["Accuracy", "SWA"], [acc, swa_test])
            plt.ylim(0, 1)
            plt.title(f"{ds_name}: Test Accuracy & SWA")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_test_metrics.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot for {ds_name}: {e}")
        plt.close()

    # --------- print evaluation metrics ---------
    if len(preds) and len(gts):
        print(
            f"{ds_name} - Test Accuracy: {(preds==gts).mean():.4f}, Test SWA: {rec['metrics'].get('test_swa', np.nan):.4f}"
        )

# --------- comparison plot across datasets ---------
try:
    if len(val_swa_curves) > 1:
        plt.figure()
        for i, (ds_name, (ep, curve)) in enumerate(val_swa_curves.items()):
            if i >= 5:  # plot at most 5 datasets
                break
            plt.plot(ep, curve, label=f"{ds_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation SWA")
        plt.title("Validation SWA Comparison Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_val_swa.png"))
        plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()
