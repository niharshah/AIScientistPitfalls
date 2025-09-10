import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- prepare output dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- plotting ----------
for ds_name, ds_dict in experiment_data.items():
    # --------- figure 1: loss curves ----------
    try:
        train_loss = [d["loss"] for d in ds_dict["losses"]["train"]]
        val_loss = [d["loss"] for d in ds_dict["losses"]["val"]]
        epochs = [d["epoch"] for d in ds_dict["losses"]["train"]]
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name}: Training vs Validation Loss\n(Left: Train, Right: Val)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # --------- figure 2: validation metrics ----------
    try:
        metrics = ds_dict["metrics"]["val"]
        epochs = [m["epoch"] for m in metrics]
        sdwa = [m["sdwa"] for m in metrics]
        cwa = [m["cwa"] for m in metrics]
        swa = [m["swa"] for m in metrics]
        plt.figure()
        plt.plot(epochs, sdwa, label="SDWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_name}: Validation Metrics Over Time\n(SDWA, CWA, SWA)")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_validation_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {ds_name}: {e}")
        plt.close()

    # --------- figure 3: class distribution ----------
    try:
        y_pred = np.array(ds_dict["predictions"])
        y_true = np.array(ds_dict["ground_truth"])
        classes = sorted(set(np.concatenate([y_true, y_pred])))
        true_counts = [np.sum(y_true == c) for c in classes]
        pred_counts = [np.sum(y_pred == c) for c in classes]

        x = np.arange(len(classes))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, true_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(f"{ds_name}: Test Set Class Distribution\n(Left: GT, Right: Pred)")
        plt.xticks(x, classes)
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_class_distribution.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating distribution plot for {ds_name}: {e}")
        plt.close()

    # ---------- print final SDWA ----------
    try:
        # SDWA on test set was printed during training loop; recompute to print again
        seqs = []  # sequence strings were not saved for test, so skip recomputation
        final_sdwa = None
        # look for metric with highest epoch as a proxy for final
        if ds_dict["metrics"]["val"]:
            final_sdwa = ds_dict["metrics"]["val"][-1]["sdwa"]
        print(f"{ds_name} final validation SDWA: {final_sdwa}")
    except Exception as e:
        print(f"Error retrieving SDWA for {ds_name}: {e}")
