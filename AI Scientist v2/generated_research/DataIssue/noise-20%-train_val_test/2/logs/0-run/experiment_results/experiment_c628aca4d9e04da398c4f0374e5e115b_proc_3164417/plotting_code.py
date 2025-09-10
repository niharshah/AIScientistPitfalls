import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# load experiment data
# -------------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = list(experiment_data.keys())
colors = plt.cm.tab10.colors

# -------------------------------------------------------------------------
# per-dataset curves
# -------------------------------------------------------------------------
for i, ds in enumerate(datasets):
    ed = experiment_data[ds]
    epochs = ed.get("epochs", [])
    tr_f1 = ed["metrics"].get("train_macro_f1", [])
    val_f1 = ed["metrics"].get("val_macro_f1", [])
    tr_ls = ed["losses"].get("train", [])
    val_ls = ed["losses"].get("val", [])

    # 1) macro-F1 curve ----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, "--", color=colors[i % 10], label="Train")
        plt.plot(epochs, val_f1, "-", color=colors[i % 10], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{ds} Macro-F1 Curves (Train dashed, Val solid)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds.lower()}_macro_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {ds}: {e}")
        plt.close()

    # 2) loss curve --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_ls, "--", color=colors[i % 10], label="Train")
        plt.plot(epochs, val_ls, "-", color=colors[i % 10], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds} Loss Curves (Train dashed, Val solid)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds.lower()}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()

    # 3) confusion matrix --------------------------------------------------
    try:
        preds = np.array(ed.get("predictions", []))
        gts = np.array(ed.get("ground_truth", []))
        if preds.size and gts.size:
            cm = confusion_matrix(gts, preds)
            plt.figure(figsize=(4, 3))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{ds} Confusion Matrix (Test Set)")
            plt.savefig(os.path.join(working_dir, f"{ds.lower()}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds}: {e}")
        plt.close()

# -------------------------------------------------------------------------
# test macro-F1 comparison bar chart (across datasets)
# -------------------------------------------------------------------------
try:
    test_scores = {
        ds: experiment_data[ds]["metrics"].get("test_macro_f1", np.nan)
        for ds in datasets
    }
    plt.figure()
    plt.bar(
        range(len(test_scores)),
        list(test_scores.values()),
        tick_label=list(test_scores.keys()),
        color=colors[: len(test_scores)],
    )
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=35, ha="right")
    plt.title("Test Macro-F1 Across Datasets")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "test_macro_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test score bar plot: {e}")
    plt.close()

# -------------------------------------------------------------------------
# numeric summary
# -------------------------------------------------------------------------
print("Test Macro-F1 summary:", {k: round(v, 4) for k, v in test_scores.items()})
