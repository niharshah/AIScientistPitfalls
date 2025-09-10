import matplotlib.pyplot as plt
import numpy as np
import os

# ensure working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper to iterate over nested dicts (only one combo expected here)
plots_done = 0
for cfg_name, cfg in experiment_data.items():
    for dname, dat in cfg.items():
        losses = dat["losses"]
        metrics = dat["metrics"]
        # 1) Loss curves ------------------------------------------------------
        try:
            plt.figure()
            plt.plot(losses["train"], label="Train")
            plt.plot(losses["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname} Training vs Validation Loss")
            plt.legend()
            fname = f"{dname}_{cfg_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plots_done += 1
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # 2) Weighted accuracy curves ----------------------------------------
        try:
            epochs = range(1, len(metrics["val"]) + 1)
            cwa = [m["CWA"] for m in metrics["val"]]
            swa = [m["SWA"] for m in metrics["val"]]
            comp = [m["CompWA"] for m in metrics["val"]]
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, comp, label="CompWA")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} Weighted Accuracies (Validation)")
            plt.legend()
            fname = f"{dname}_{cfg_name}_weighted_acc.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plots_done += 1
        except Exception as e:
            print(f"Error creating accuracy plot: {e}")
            plt.close()

        # 3) Final test metrics bar chart ------------------------------------
        try:
            test_m = metrics["test"]
            names = list(test_m.keys())
            vals = list(test_m.values())
            plt.figure()
            plt.bar(names, vals)
            plt.ylim(0, 1)
            for i, v in enumerate(vals):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            plt.title(f"{dname} Test Weighted Accuracies")
            fname = f"{dname}_{cfg_name}_test_metrics.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plots_done += 1
        except Exception as e:
            print(f"Error creating test metric bar plot: {e}")
            plt.close()

        # 4) Confusion matrix heat-map ---------------------------------------
        try:
            gtruth = dat["ground_truth"]
            preds = dat["predictions"]
            num_cls = max(gtruth + preds) + 1 if gtruth else 0
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gtruth, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{dname} Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            fname = f"{dname}_{cfg_name}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plots_done += 1
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

        # print final test metrics
        print(f"{dname} final test metrics:", metrics.get("test", {}))

print(f"Total plots saved: {plots_done}")
