import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dataset_name, runs in experiment_data.items():
    models = list(runs.keys())
    epochs, loss_tr, loss_val, f1_tr, f1_val, final_val_f1 = {}, {}, {}, {}, {}, {}
    for m in models:
        loss_tr[m] = [x["loss"] for x in runs[m]["losses"]["train"]]
        loss_val[m] = [x["loss"] for x in runs[m]["losses"]["val"]]
        f1_tr[m] = [x["macro_F1"] for x in runs[m]["metrics"]["train"]]
        f1_val[m] = [x["macro_F1"] for x in runs[m]["metrics"]["val"]]
        epochs[m] = [x["epoch"] for x in runs[m]["metrics"]["train"]]
        final_val_f1[m] = f1_val[m][-1] if f1_val[m] else 0.0

    # 1) Loss curves ------------------------------------------------
    try:
        plt.figure()
        for m in models:
            plt.plot(epochs[m], loss_tr[m], "--", label=f"{m}-train")
            plt.plot(epochs[m], loss_val[m], label=f"{m}-val")
        plt.xlabel("Epoch")
        plt.ylabel("CE Loss")
        plt.title(f"Loss Curves\nDataset: {dataset_name} (Train vs Val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dataset_name}: {e}")
        plt.close()

    # 2) Macro-F1 curves -------------------------------------------
    try:
        plt.figure()
        for m in models:
            plt.plot(epochs[m], f1_tr[m], "--", label=f"{m}-train")
            plt.plot(epochs[m], f1_val[m], label=f"{m}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"Macro-F1 Curves\nDataset: {dataset_name} (Train vs Val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves for {dataset_name}: {e}")
        plt.close()

    # 3) Final validation F1 bar chart ------------------------------
    try:
        plt.figure()
        xs = np.arange(len(models))
        vals = [final_val_f1[m] for m in models]
        plt.bar(xs, vals, tick_label=models)
        plt.xlabel("Model")
        plt.ylabel("Final Val Macro-F1")
        plt.title(f"Final Validation Macro-F1 Comparison\nDataset: {dataset_name}")
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_final_val_f1_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart for {dataset_name}: {e}")
        plt.close()

    # 4) Confusion matrix for best model ---------------------------
    try:
        best_model = max(final_val_f1, key=final_val_f1.get)
        preds = runs[best_model].get("predictions", [])
        gts = runs[best_model].get("ground_truth", [])
        if preds and gts:
            cm = confusion_matrix(gts, preds, labels=np.unique(gts))
            disp = ConfusionMatrixDisplay(cm)
            plt.figure()
            disp.plot(values_format="d", cmap="Blues", colorbar=False)
            plt.title(
                f"Confusion Matrix (Val)\nDataset: {dataset_name} – Best: {best_model}"
            )
            plt.savefig(
                os.path.join(
                    working_dir, f"{dataset_name}_{best_model}_conf_matrix.png"
                )
            )
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dataset_name}: {e}")
        plt.close()
