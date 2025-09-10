import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_val_overall = {}

for dname, d in experiment_data.items():
    print(
        f"{dname}: Test {d.get('primary_metric','Macro-F1')} = {d.get('test_macroF1','N/A'):.4f}"
    )
    epochs = d.get("epochs", [])
    tr_loss = d.get("losses", {}).get("train", [])
    val_loss = d.get("losses", {}).get("val", [])
    tr_met = d.get("metrics", {}).get("train", [])
    val_met = d.get("metrics", {}).get("val", [])
    preds = d.get("predictions", [])
    gts = d.get("ground_truth", [])
    best_val_overall[dname] = max(val_met) if val_met else np.nan

    # -------- loss curve --------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname}: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dname}: {e}")
        plt.close()

    # -------- metric curve -------
    try:
        plt.figure()
        plt.plot(epochs, tr_met, label="Train Macro-F1")
        plt.plot(epochs, val_met, label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dname}: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_metric_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting metric for {dname}: {e}")
        plt.close()

    # -------- confusion matrix ---
    try:
        if preds and gts:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                f"{dname}: Normalized Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dname}: {e}")
        plt.close()

# -------- comparison plot across datasets ----------
try:
    if len(best_val_overall) > 1:
        plt.figure()
        names = list(best_val_overall.keys())
        scores = [best_val_overall[n] for n in names]
        plt.bar(names, scores)
        plt.ylabel("Best Validation Macro-F1")
        plt.title("Dataset Comparison: Best Val Macro-F1")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "datasets_best_val_metric.png"))
        plt.close()
except Exception as e:
    print(f"Error plotting dataset comparison: {e}")
    plt.close()
