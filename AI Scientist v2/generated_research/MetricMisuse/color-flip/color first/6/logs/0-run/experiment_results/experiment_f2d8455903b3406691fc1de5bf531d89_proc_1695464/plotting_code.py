import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ setup ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Iterate over every dataset contained in experiment_data
for ds_name, ds_dict in experiment_data.items():
    # -------- plot 1: loss curves ------------------------
    try:
        tr_epochs, tr_loss = zip(*ds_dict["losses"]["train"])
        va_epochs, va_loss = zip(*ds_dict["losses"]["val"])
        plt.figure()
        plt.plot(tr_epochs, tr_loss, label="Train")
        plt.plot(va_epochs, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Loss Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = f"loss_curve_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # -------- plot 2: metric evolution --------------------
    try:
        # metrics stored as list of (epoch, dict)
        metr_hist = ds_dict["metrics"]["val"]
        epochs = [e for e, _ in metr_hist]
        cwa = [d["CWA"] for _, d in metr_hist]
        swa = [d["SWA"] for _, d in metr_hist]
        pcwa = [d["PCWA"] for _, d in metr_hist]

        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, pcwa, label="PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_name} Validation Metrics Across Epochs")
        plt.legend()
        fname = f"metric_curves_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting metrics for {ds_name}: {e}")
        plt.close()

    # -------- plot 3: final metric bar chart --------------
    try:
        last_ep, last_vals = ds_dict["metrics"]["val"][-1]
        metrics = ["CWA", "SWA", "PCWA"]
        vals = [last_vals[m] for m in metrics]
        x = np.arange(len(metrics))

        plt.figure()
        plt.bar(x, vals, color=["tab:blue", "tab:orange", "tab:green"])
        plt.xticks(x, metrics)
        plt.ylabel("Score")
        plt.title(f"{ds_name} Final-Epoch Validation Metrics (epoch {last_ep})")
        fname = f"final_val_metrics_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating summary bar chart for {ds_name}: {e}")
        plt.close()

    # -------- print final test metrics -------------------
    try:
        y_true = ds_dict.get("ground_truth", [])
        y_pred = ds_dict.get("predictions", [])
        if y_true and y_pred:
            # compute already stored test metrics by reusing last validation metric code
            # they were printed during training but not stored, so we approximate with
            # supplied val metrics if real test not present
            print(f"{ds_name} - test predictions available: {len(y_pred)} samples")
        else:
            print(f"{ds_name} - no test predictions stored")
    except Exception as e:
        print(f"Error printing test metrics for {ds_name}: {e}")
