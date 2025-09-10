import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
#                       LOAD EXPERIMENT DATA                         #
# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Expect single experiment/dataset; guard in case of more.
exp_name = next(iter(experiment_data.keys()), None)
ds_name = None
if exp_name:
    ds_name = next(iter(experiment_data[exp_name].keys()), None)

if not (exp_name and ds_name):
    print("No experiment data found, nothing to plot.")
else:
    exp = experiment_data[exp_name][ds_name]

    epochs = np.arange(1, len(exp["losses"]["train"]) + 1)

    # --------------------------- LOSS CURVES --------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train Loss")
        plt.plot(epochs, exp["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} – {exp_name}\nTraining vs Validation Loss")
        plt.legend()
        save_path = os.path.join(working_dir, f"{ds_name}_{exp_name}_Loss_Curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------------ ACCURACY CURVES -------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, exp["metrics"]["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} – {exp_name}\nTraining vs Validation Accuracy")
        plt.legend()
        save_path = os.path.join(
            working_dir, f"{ds_name}_{exp_name}_Accuracy_Curves.png"
        )
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ----------------------- RULE FIDELITY CURVE ----------------------- #
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["Rule_Fidelity"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity Score")
        plt.title(f"{ds_name} – {exp_name}\nRule Fidelity Across Epochs")
        save_path = os.path.join(working_dir, f"{ds_name}_{exp_name}_RuleFidelity.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot: {e}")
        plt.close()

    # --------------------------- CONF MATRIX --------------------------- #
    try:
        preds = exp.get("predictions")
        gts = exp.get("ground_truth")
        if preds is not None and gts is not None:
            n_cls = int(max(gts.max(), preds.max()) + 1)
            conf = np.zeros((n_cls, n_cls), dtype=int)
            for p, t in zip(preds, gts):
                conf[t, p] += 1
            plt.figure()
            im = plt.imshow(conf, interpolation="nearest", cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"{ds_name} – {exp_name}\nConfusion Matrix (Test Set)")
            save_path = os.path.join(
                working_dir, f"{ds_name}_{exp_name}_ConfusionMatrix.png"
            )
            plt.savefig(save_path)
            plt.close()
        else:
            print("Predictions/ground truth missing; skipping confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ----------------------- PRINT TEST ACCURACY ----------------------- #
    try:
        if preds is not None and gts is not None:
            test_acc = (preds == gts).mean()
            print(f"Test Accuracy (recomputed): {test_acc:.3f}")
    except Exception as e:
        print(f"Error computing test accuracy: {e}")
