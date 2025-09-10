import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

summary = {}
if experiment_data is not None:
    for exp_key, exp_val in experiment_data.get("epochs_tuning", {}).items():
        run = exp_val.get("SPR_BENCH", {})
        losses_tr = [x["loss"] for x in run.get("losses", {}).get("train", [])]
        losses_val = [x["loss"] for x in run.get("losses", {}).get("val", [])]
        f1_tr = [x["macro_f1"] for x in run.get("metrics", {}).get("train", [])]
        f1_val = [x["macro_f1"] for x in run.get("metrics", {}).get("val", [])]
        epochs = list(range(1, len(losses_tr) + 1))
        # store last val F1 for summary
        if f1_val:
            summary[exp_key] = f1_val[-1]

        try:
            plt.figure(figsize=(10, 4))
            # Left subplot: Loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, losses_tr, label="Train Loss")
            plt.plot(epochs, losses_val, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Left: Loss")

            # Right subplot: F1
            plt.subplot(1, 2, 2)
            plt.plot(epochs, f1_tr, label="Train F1")
            plt.plot(epochs, f1_val, label="Val F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro F1")
            plt.legend()
            plt.title("Right: Macro F1")

            plt.suptitle(
                f"SPR_BENCH Results for {exp_key}\nLeft: Loss Curves, Right: F1 Curves"
            )
            save_name = f"SPR_BENCH_{exp_key}_loss_f1_curves.png"
            plt.savefig(os.path.join(working_dir, save_name))
            plt.close()
        except Exception as e:
            print(f"Error creating plot for {exp_key}: {e}")
            plt.close()

# print summary table
if summary:
    print("Final Validation Macro F1 per Epoch Budget:")
    for k, v in sorted(
        summary.items(), key=lambda x: int(x[0][1:])
    ):  # sort by numeric epochs
        print(f"  {k}: {v:.4f}")
