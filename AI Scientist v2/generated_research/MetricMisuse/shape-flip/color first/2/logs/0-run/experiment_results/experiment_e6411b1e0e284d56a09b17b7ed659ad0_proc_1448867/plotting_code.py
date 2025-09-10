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

# Identify datasets (ignore any 'pooling_type' wrapper)
datasets = [k for k in experiment_data.keys() if k != "pooling_type"]

for ds in datasets:
    ds_data = experiment_data.get(ds, {})
    # ---- Loss curves -------------------------------------------------------
    try:
        tr_losses = sorted(ds_data["losses"]["train"])
        val_losses = sorted(ds_data["losses"]["val"])
        ep_tr, v_tr = zip(*tr_losses)
        ep_val, v_val = zip(*val_losses)

        plt.figure(figsize=(7, 4))
        plt.plot(ep_tr, v_tr, "--o", label="Train")
        plt.plot(ep_val, v_val, "-o", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds} – Training and Validation Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {ds}: {e}")
        plt.close()

    # ---- PCWA curves --------------------------------------------------------
    try:
        tr_pcwa = sorted(ds_data["metrics"]["train"])
        val_pcwa = sorted(ds_data["metrics"]["val"])
        ep_tr, m_tr = zip(*tr_pcwa)
        ep_val, m_val = zip(*val_pcwa)

        plt.figure(figsize=(7, 4))
        plt.plot(ep_tr, m_tr, "--o", label="Train PCWA")
        plt.plot(ep_val, m_val, "-o", label="Val PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.title(f"{ds} – Training and Validation PCWA Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_pcwa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PCWA curves for {ds}: {e}")
        plt.close()

    # ---- Final PCWA bar chart ----------------------------------------------
    try:
        final_train_pcwa = tr_pcwa[-1][1]
        final_val_pcwa = val_pcwa[-1][1]
        plt.figure(figsize=(5, 4))
        plt.bar(
            ["Train", "Validation"],
            [final_train_pcwa, final_val_pcwa],
            color=["steelblue", "orange"],
        )
        plt.ylabel("Final PCWA")
        plt.title(f"{ds} – Final PCWA Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_final_pcwa_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating final PCWA bar for {ds}: {e}")
        plt.close()

    # ---- Prediction vs Ground-Truth distribution ---------------------------
    try:
        preds = np.array(ds_data.get("predictions", []))
        gts = np.array(ds_data.get("ground_truth", []))
        if preds.size and gts.size:
            classes = sorted(set(gts) | set(preds))
            gt_counts = [(gts == c).sum() for c in classes]
            pr_counts = [(preds == c).sum() for c in classes]

            x = np.arange(len(classes))
            width = 0.35
            plt.figure(figsize=(6, 4))
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pr_counts, width, label="Predictions")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(
                f"{ds} – Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.xticks(x, classes)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds}_class_distribution.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating class distribution for {ds}: {e}")
        plt.close()
