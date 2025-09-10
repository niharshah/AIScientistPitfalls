import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    for ds_name, ds in experiment_data.items():
        pre_ls = np.array(ds["losses"].get("pretrain", []), dtype=float)
        tr_ls = np.array(ds["losses"].get("train", []), dtype=float)
        val_ls = np.array(ds["losses"].get("val", []), dtype=float)
        swa_arr = np.array(ds["metrics"].get("SWA", []), dtype=float)
        cwa_arr = np.array(ds["metrics"].get("CWA", []), dtype=float)
        scwa_arr = np.array(ds["metrics"].get("SCWA", []), dtype=float)
        ep_idx_pre = np.arange(1, len(pre_ls) + 1)
        ep_idx_ft = np.arange(1, len(tr_ls) + 1)

        # -------------- Plot 1: pre-training loss -----------------
        try:
            if pre_ls.size:
                plt.figure()
                plt.plot(ep_idx_pre, pre_ls, marker="o")
                plt.xlabel("Epoch")
                plt.ylabel("NT-Xent Loss")
                plt.title(f"{ds_name} Pre-Training Loss Curve")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_pretrain_loss.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating {ds_name} pretrain loss: {e}")
            plt.close()

        # -------------- Plot 2: fine-tuning loss ------------------
        try:
            if tr_ls.size and val_ls.size:
                plt.figure()
                plt.plot(ep_idx_ft, tr_ls, label="Train CE Loss")
                plt.plot(ep_idx_ft, val_ls, label="Val CE Loss")
                plt.xlabel("Fine-Tune Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{ds_name} Fine-Tuning Loss\nLeft: Train, Right: Val")
                plt.legend()
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_finetune_loss.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating {ds_name} fine-tune loss: {e}")
            plt.close()

        # -------------- Plot 3: metric curves ---------------------
        try:
            if scwa_arr.size:
                plt.figure()
                plt.plot(ep_idx_ft, swa_arr, label="SWA")
                plt.plot(ep_idx_ft, cwa_arr, label="CWA")
                plt.plot(ep_idx_ft, scwa_arr, label="SCWA")
                plt.xlabel("Fine-Tune Epoch")
                plt.ylabel("Weighted Accuracy")
                plt.title(f"{ds_name} Accuracy Metrics over Epochs")
                plt.legend()
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_metric_curves.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating {ds_name} metric curves: {e}")
            plt.close()

        # -------------- Plot 4: final metric bar ------------------
        try:
            if scwa_arr.size:
                finals = [swa_arr[-1], cwa_arr[-1], scwa_arr[-1]]
                labels = ["SWA", "CWA", "SCWA"]
                plt.figure()
                plt.bar(labels, finals, color=["#66c2a5", "#fc8d62", "#8da0cb"])
                plt.ylabel("Final Score")
                plt.title(f"{ds_name} Final Validation Metrics")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_final_metrics.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating {ds_name} final metric bar: {e}")
            plt.close()

        # -------------- print best epoch --------------------------
        if scwa_arr.size:
            best_ep = scwa_arr.argmax() + 1
            print(
                f"{ds_name}: best SCWA={scwa_arr.max():.4f} at epoch {best_ep} "
                f"(SWA={swa_arr[best_ep-1]:.4f}, CWA={cwa_arr[best_ep-1]:.4f})"
            )
