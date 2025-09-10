import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------- load experiment results ----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------- iterate over ablations / datasets ----------------------
for ablation, ds_dict in experiment_data.items():
    for ds_name, rec in ds_dict.items():
        # ---------- print final test metrics ----------
        if "metrics" in rec and "test" in rec["metrics"]:
            lr, cwa, swa, hwa, cna = rec["metrics"]["test"]
            print(
                f"{ablation} | {ds_name} | TEST  CWA={cwa:.3f}  SWA={swa:.3f}  "
                f"HWA={hwa:.3f}  CNA={cna:.3f}"
            )

        # ---------- 1) loss curves ----------
        try:
            plt.figure()
            tr = rec.get("losses", {}).get("train", [])
            va = rec.get("losses", {}).get("val", [])
            if tr:
                epochs_tr = [t[1] for t in tr]
                losses_tr = [t[2] for t in tr]
                plt.plot(epochs_tr, losses_tr, label="train")
            if va:
                epochs_va = [v[1] for v in va]
                losses_va = [v[2] for v in va]
                plt.plot(epochs_va, losses_va, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ablation} – {ds_name} Loss Curve")
            plt.legend()
            fname = f"{ablation}_{ds_name}_loss_curve.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {ds_name}: {e}")
            plt.close()

        # ---------- 2) validation metric curves ----------
        try:
            m = rec.get("metrics", {}).get("val", [])
            if m:
                plt.figure()
                ep = [x[1] for x in m]
                plt.plot(ep, [x[2] for x in m], label="CWA")
                plt.plot(ep, [x[3] for x in m], label="SWA")
                plt.plot(ep, [x[4] for x in m], label="HWA")
                plt.plot(ep, [x[5] for x in m], label="CNA")
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.title(f"{ablation} – {ds_name} Validation Metrics")
                plt.legend()
                fname = f"{ablation}_{ds_name}_metric_curves.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating metric curves for {ds_name}: {e}")
            plt.close()

        # ---------- 3) confusion matrix ----------
        try:
            preds = np.array(rec.get("predictions", []))
            gts = np.array(rec.get("ground_truth", []))
            if preds.size and gts.size:
                labels = sorted(set(gts) | set(preds))
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for yt, yp in zip(gts, preds):
                    cm[labels.index(yt), labels.index(yp)] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xticks(range(len(labels)), labels)
                plt.yticks(range(len(labels)), labels)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(
                    f"{ablation} – {ds_name} Confusion Matrix\n"
                    "Left: Ground Truth, Right: Predicted"
                )
                fname = f"{ablation}_{ds_name}_conf_matrix.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()
