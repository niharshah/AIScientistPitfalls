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
    experiment_data = {}

for dataset_name, tasks in experiment_data.items():
    for task_name, ed in tasks.items():
        ep = ed.get("epochs", [])
        tl = ed.get("losses", {}).get("train", [])
        vl = ed.get("losses", {}).get("val", [])
        thwa = [m.get("HWA") for m in ed.get("metrics", {}).get("train", [])]
        vhwa = [m.get("HWA") for m in ed.get("metrics", {}).get("val", [])]
        best_ep = ed.get("best_epoch")

        # --------- Plot 1: Loss curves ---------------------------------------
        try:
            plt.figure()
            plt.plot(ep, tl, label="Train")
            plt.plot(ep, vl, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset_name}-{task_name}: Training vs Validation Loss")
            plt.legend()
            fname = f"{dataset_name}_{task_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot ({dataset_name}-{task_name}): {e}")
            plt.close()

        # --------- Plot 2: HWA curves ----------------------------------------
        try:
            plt.figure()
            plt.plot(ep, thwa, label="Train HWA")
            plt.plot(ep, vhwa, label="Val HWA")
            if best_ep is not None:
                plt.axvline(
                    best_ep, color="r", linestyle="--", label=f"Best Epoch {best_ep}"
                )
            plt.xlabel("Epoch")
            plt.ylabel("HWA")
            plt.title(f"{dataset_name}-{task_name}: Training vs Validation HWA")
            plt.legend()
            fname = f"{dataset_name}_{task_name}_HWA_curves.png"
            plt.savefig(os.path.join(working_dir, fname), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating HWA plot ({dataset_name}-{task_name}): {e}")
            plt.close()

        # --------- Plot 3: Test label distribution ---------------------------
        try:
            preds = ed.get("predictions")
            gts = ed.get("ground_truth")
            if preds is not None and gts is not None and len(preds) == len(gts) > 0:
                plt.figure()
                labels = sorted(set(gts + preds))
                gt_cnt = [gts.count(l) for l in labels]
                pr_cnt = [preds.count(l) for l in labels]
                x = np.arange(len(labels))
                w = 0.35
                plt.bar(x - w / 2, gt_cnt, width=w, label="Ground Truth")
                plt.bar(x + w / 2, pr_cnt, width=w, label="Predictions")
                plt.xticks(x, labels)
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title(f"{dataset_name}-{task_name}: Test Label Distribution")
                plt.legend()
                fname = f"{dataset_name}_{task_name}_test_label_dist.png"
                plt.savefig(os.path.join(working_dir, fname), dpi=150)
                plt.close()
        except Exception as e:
            print(f"Error creating distribution plot ({dataset_name}-{task_name}): {e}")
            plt.close()

        # Print best validation HWA
        if best_ep is not None and 0 < best_ep <= len(vhwa):
            print(
                f"{dataset_name}-{task_name}: best HWA={vhwa[best_ep-1]:.4f} at epoch {best_ep}"
            )
