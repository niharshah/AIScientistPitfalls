import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- SETUP -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- LOAD EXPERIMENT ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------------- PLOTTING ----------------
for variant, v_dict in experiment_data.items():
    for dataset, d_dict in v_dict.items():
        # 1) Loss curves -------------------------------------------------------
        try:
            train_loss = d_dict.get("losses", {}).get("train", [])
            val_loss = d_dict.get("losses", {}).get("val", [])
            if train_loss and val_loss:
                plt.figure()
                epochs = range(1, len(train_loss) + 1)
                plt.plot(epochs, train_loss, label="Train Loss")
                plt.plot(epochs, val_loss, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                title = f"{dataset} – {variant}\nTraining vs Validation Loss"
                plt.title(title)
                plt.legend()
                fname = f"{dataset}_{variant}_loss_curve.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dataset}/{variant}: {e}")
            plt.close()

        # 2) Validation metric curves -----------------------------------------
        try:
            val_metrics = d_dict.get("metrics", {}).get("val", [])
            if val_metrics:
                epochs = [m["epoch"] for m in val_metrics]
                cwa = [m["cwa"] for m in val_metrics]
                swa = [m["swa"] for m in val_metrics]
                cpx = [m["cpxwa"] for m in val_metrics]
                plt.figure()
                plt.plot(epochs, cwa, label="CWA")
                plt.plot(epochs, swa, label="SWA")
                plt.plot(epochs, cpx, label="CpxWA")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                title = f"{dataset} – {variant}\nValidation Weighted Accuracies"
                plt.title(title)
                plt.legend()
                fname = f"{dataset}_{variant}_val_metrics.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating metric curve for {dataset}/{variant}: {e}")
            plt.close()

        # ---------------- PRINT TEST METRICS -----------------
        test_metrics = d_dict.get("metrics", {}).get("test", {})
        if test_metrics:
            print(
                f"{dataset}/{variant} TEST  "
                f"CWA={test_metrics.get('cwa', 'NA'):.3f}  "
                f"SWA={test_metrics.get('swa', 'NA'):.3f}  "
                f"CpxWA={test_metrics.get('cpxwa', 'NA'):.3f}"
            )
