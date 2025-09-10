import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

final_scores = {}

# --------- Load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# --------- Iterate over datasets ----------
for dset_name, dset in experiment_data.items():
    losses = dset.get("losses", {})
    val_metrics = dset.get("metrics", {}).get("val", [])
    epochs = range(1, len(losses.get("train", [])) + 1)

    # ---- Plot loss curve ----
    try:
        plt.figure()
        if losses.get("train"):
            plt.plot(epochs, losses["train"], label="Train")
        if losses.get("val"):
            plt.plot(epochs, losses["val"], label="Validation")
        plt.title(f"{dset_name} – Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"{dset_name.lower()}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset_name}: {e}")
        plt.close()

    # ---- Plot weighted accuracies ----
    try:
        if val_metrics:
            cwa = [m.get("cwa", np.nan) for m in val_metrics]
            swa = [m.get("swa", np.nan) for m in val_metrics]
            ewa = [m.get("ewa", np.nan) for m in val_metrics]

            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, ewa, label="EWA")
            plt.title(f"{dset_name} – Weighted Accuracies")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = f"{dset_name.lower()}_weighted_accuracies.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()

            # store final epoch scores
            final_scores[dset_name] = {"CWA": cwa[-1], "SWA": swa[-1], "EWA": ewa[-1]}
    except Exception as e:
        print(f"Error creating metric plot for {dset_name}: {e}")
        plt.close()

# --------- Print final metrics ----------
for dset, scores in final_scores.items():
    print(
        f"{dset} – Final Metrics: "
        f"CWA={scores['CWA']:.3f}, "
        f"SWA={scores['SWA']:.3f}, "
        f"EWA={scores['EWA']:.3f}"
    )
