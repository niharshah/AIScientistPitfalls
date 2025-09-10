import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    # Only dataset present
    ds_name = "SPR_transformer"
    if ds_name not in experiment_data:
        print(f"{ds_name} not found in experiment_data")
    else:
        data = experiment_data[ds_name]

        pre_losses = np.array(data["losses"].get("pretrain", []))
        tr_losses = np.array(data["losses"].get("train", []))
        val_losses = np.array(data["losses"].get("val", []))
        swa = np.array(data["metrics"].get("val_SWA", []))
        cwa = np.array(data["metrics"].get("val_CWA", []))
        scwa_vals = np.array(data["metrics"].get("val_SCWA", []))
        preds = np.array(data.get("predictions", []))
        gts = np.array(data.get("ground_truth", []))

        # ------------------ Plot 1: Pre-training loss ---------------
        try:
            if pre_losses.size:
                plt.figure()
                plt.plot(np.arange(1, len(pre_losses) + 1), pre_losses, marker="o")
                plt.xlabel("Epoch")
                plt.ylabel("NT-Xent Loss")
                plt.title(f"{ds_name} Pre-training Loss\nLeft: Loss vs Epoch")
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_pretrain_loss.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating pretraining loss plot: {e}")
            plt.close()

        # ------------------ Plot 2: Fine-tune losses ---------------
        try:
            if tr_losses.size and val_losses.size:
                epochs = np.arange(1, len(tr_losses) + 1)
                plt.figure()
                plt.plot(epochs, tr_losses, label="Train Loss")
                plt.plot(epochs, val_losses, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{ds_name} Fine-tune Losses\nLeft: Train, Right: Val")
                plt.legend()
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_finetune_losses.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating fine-tune loss plot: {e}")
            plt.close()

        # ------------------ Plot 3: Validation metrics -------------
        try:
            if scwa_vals.size:
                epochs = np.arange(1, len(scwa_vals) + 1)
                plt.figure()
                plt.plot(epochs, swa, label="SWA")
                plt.plot(epochs, cwa, label="CWA")
                plt.plot(epochs, scwa_vals, label="SCWA")
                plt.xlabel("Epoch")
                plt.ylabel("Metric Value")
                plt.title(
                    f"{ds_name} Validation Metrics\nLeft: SWA, Mid: CWA, Right: SCWA"
                )
                plt.legend()
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_val_metrics.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating validation metric plot: {e}")
            plt.close()

        # ------------------ Plot 4: Confusion matrix ---------------
        try:
            if preds.size and gts.size:
                num_lbl = int(max(preds.max(), gts.max())) + 1
                cm = np.zeros((num_lbl, num_lbl), dtype=int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                plt.figure(figsize=(6, 5))
                im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(
                    f"{ds_name} Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
                )
                plt.tight_layout()
                fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix plot: {e}")
            plt.close()

        # ------------------ Print summary metrics ------------------
        if scwa_vals.size:
            best_idx = scwa_vals.argmax()
            print(
                f"Best epoch={best_idx+1} | SCWA={scwa_vals[best_idx]:.4f} | "
                f"SWA={swa[best_idx]:.4f} | CWA={cwa[best_idx]:.4f}"
            )
