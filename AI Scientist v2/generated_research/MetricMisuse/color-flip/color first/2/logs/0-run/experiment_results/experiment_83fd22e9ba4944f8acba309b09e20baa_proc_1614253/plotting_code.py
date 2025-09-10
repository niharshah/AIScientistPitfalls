import matplotlib.pyplot as plt
import numpy as np
import os

# ------------ paths & data ------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------ iterate over models / datasets ------------
for model_name, ds_dict in experiment_data.items():
    for ds_name, rec in ds_dict.items():
        losses = rec.get("losses", {})
        metrics = rec.get("metrics", {})
        # ------------- Plot 1: loss curve -------------
        try:
            train_loss = losses.get("train", [])
            val_loss = losses.get("val", [])
            if train_loss or val_loss:
                plt.figure()
                if train_loss:
                    plt.plot(train_loss, label="Train")
                if val_loss:
                    plt.plot(val_loss, label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{model_name} on {ds_name}\nTraining vs Validation Loss")
                plt.legend()
                fname = f"{ds_name}_{model_name}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating loss curve: {e}")
            plt.close()

        # ------------- Plot 2: validation CWA -------------
        try:
            val_metrics = metrics.get("val", [])
            cwa_vals = [m.get("CWA") for m in val_metrics if m]
            if cwa_vals:
                plt.figure()
                plt.plot(cwa_vals, marker="o")
                plt.ylim(0, 1)
                plt.xlabel("Epoch")
                plt.ylabel("Color-Weighted Accuracy")
                plt.title(f"{model_name} on {ds_name}\nValidation CWA Across Epochs")
                fname = f"{ds_name}_{model_name}_val_CWA_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating CWA curve: {e}")
            plt.close()

        # ------------- Plot 3: test metric bars -------------
        try:
            test_metrics = metrics.get("test", {})
            if test_metrics:
                labels = list(test_metrics.keys())
                values = [test_metrics[k] for k in labels]
                plt.figure()
                plt.bar(labels, values, color="skyblue")
                plt.ylim(0, 1)
                plt.title(f"{model_name} on {ds_name}\nTest Metrics")
                for i, v in enumerate(values):
                    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
                fname = f"{ds_name}_{model_name}_test_metrics.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating test metric bar chart: {e}")
            plt.close()

        # ------------- Print test metrics -------------
        if metrics.get("test"):
            print(f"{model_name} | {ds_name} test metrics:", metrics["test"])
