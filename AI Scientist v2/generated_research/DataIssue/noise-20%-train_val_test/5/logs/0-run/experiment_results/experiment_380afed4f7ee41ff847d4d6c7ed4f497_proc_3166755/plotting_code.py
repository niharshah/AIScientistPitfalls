import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- SET-UP -------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

test_f1_summary = {}

# ----------------- PER-DATASET PLOTS ---------------- #
for ds_idx, (ds_name, ds_dict) in enumerate(experiment_data.items()):
    # -------- F1 curves -------- #
    try:
        train_f1 = ds_dict["metrics"].get("train_f1", [])
        val_f1 = ds_dict["metrics"].get("val_f1", [])
        if train_f1 and val_f1:
            epochs = np.arange(1, len(train_f1) + 1)
            plt.figure()
            plt.plot(epochs, train_f1, marker="o", label="Train F1")
            plt.plot(epochs, val_f1, marker="x", label="Val F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{ds_name} F1 Curves")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_f1_curves.png")
            plt.savefig(fname)
            print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting F1 curves for {ds_name}: {e}")
        plt.close()

    # -------- Loss curves -------- #
    try:
        tr_loss = ds_dict["losses"].get("train", [])
        val_loss = ds_dict["losses"].get("val", [])
        if tr_loss and val_loss:
            epochs = np.arange(1, len(tr_loss) + 1)
            plt.figure()
            plt.plot(epochs, tr_loss, marker="o", label="Train Loss")
            plt.plot(epochs, val_loss, marker="x", label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} Loss Curves")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
            plt.savefig(fname)
            print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting Loss curves for {ds_name}: {e}")
        plt.close()

    # -------- Confusion matrix -------- #
    try:
        if ds_idx < 5:  # plot at most 5 confusion matrices
            preds = np.asarray(ds_dict.get("predictions", []))
            gts = np.asarray(ds_dict.get("ground_truth", []))
            if preds.size and gts.size:
                num_cls = len(np.unique(gts))
                cm = np.zeros((num_cls, num_cls), dtype=int)
                for p, g in zip(preds, gts):
                    cm[g, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{ds_name} Confusion Matrix")
                for i in range(num_cls):
                    for j in range(num_cls):
                        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
                fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
                plt.savefig(fname)
                print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting Confusion Matrix for {ds_name}: {e}")
        plt.close()

    # collect test F1 for summary / comparison
    test_f1_summary[ds_name] = ds_dict["metrics"].get("test_f1", None)

# ----------------- CROSS-DATASET BAR CHART ---------------- #
try:
    if len(test_f1_summary) > 1:
        plt.figure()
        names = list(test_f1_summary.keys())
        vals = [test_f1_summary[n] for n in names]
        plt.bar(names, vals, color="skyblue")
        plt.ylabel("Test Macro-F1")
        plt.title("Test Macro-F1 across Datasets")
        fname = os.path.join(working_dir, "datasets_test_f1_comparison.png")
        plt.savefig(fname)
        print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error plotting cross-dataset comparison: {e}")
    plt.close()

# ----------------- PRINT SUMMARY ---------------- #
for k, v in test_f1_summary.items():
    print(f"{k}: Test macro-F1 = {v}")
