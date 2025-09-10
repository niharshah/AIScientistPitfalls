import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

saved = []
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for exp_name, datasets in experiment_data.items():
    for ds_name, content in datasets.items():
        losses = content["losses"]
        metrics = content["metrics"]
        preds = np.array(content.get("predictions", []))
        gts = np.array(content.get("ground_truth", []))
        # ---- 1: loss curves ----
        try:
            plt.figure()
            epochs = range(1, len(losses["train"]) + 1)
            plt.plot(epochs, losses["train"], label="Train")
            plt.plot(epochs, losses["val"], label="Validation")
            plt.title(f"{ds_name} – Loss Curves ({exp_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fn = os.path.join(working_dir, f"{ds_name}_{exp_name}_loss_curves.png")
            plt.savefig(fn)
            saved.append(fn)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()
        # ---- 2: metric curves ----
        try:
            plt.figure()
            macro_f1 = [m["macro_f1"] for m in metrics["val"]]
            cwa = [m["cwa"] for m in metrics["val"]]
            plt.plot(epochs, macro_f1, label="Macro-F1")
            plt.plot(epochs, cwa, label="CWA")
            plt.title(f"{ds_name} – Validation Metrics ({exp_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            fn = os.path.join(working_dir, f"{ds_name}_{exp_name}_metrics_curves.png")
            plt.savefig(fn)
            saved.append(fn)
            plt.close()
        except Exception as e:
            print(f"Error creating metric plot: {e}")
            plt.close()
        # ---- 3: prediction scatter ----
        try:
            if preds.size and gts.size:
                idx = np.linspace(0, len(preds) - 1, num=min(200, len(preds))).astype(
                    int
                )
                plt.figure()
                plt.scatter(gts[idx], preds[idx], alpha=0.6, s=10)
                plt.title(
                    f"Ground Truth vs Predictions – {ds_name} ({exp_name})\nLeft: Ground Truth, Right: Generated Samples"
                )
                plt.xlabel("Ground Truth")
                plt.ylabel("Predictions")
                fn = os.path.join(working_dir, f"{ds_name}_{exp_name}_gt_vs_pred.png")
                plt.savefig(fn)
                saved.append(fn)
                plt.close()
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            plt.close()
        # ---- print last-epoch metrics ----
        if metrics["val"]:
            last = metrics["val"][-1]
            print(
                f"{exp_name}/{ds_name} – final Macro-F1: {last['macro_f1']:.3f}, CWA: {last['cwa']:.3f}"
            )

print("Saved figures:")
for s in saved:
    print("  ", s)
