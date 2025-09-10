import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# -------- iterate over all stored runs --------
for model_name, datasets in experiment_data.items():
    for ds_name, data in datasets.items():
        epochs = data.get("epochs", [])
        losses_tr = data.get("losses", {}).get("train", [])
        losses_val = data.get("losses", {}).get("val", [])
        f1_tr = data.get("metrics", {}).get("train_f1", [])
        f1_val = data.get("metrics", {}).get("val_f1", [])
        preds = np.array(data.get("predictions", []))
        gts = np.array(data.get("ground_truth", []))

        # ---- 1) loss curves ----
        try:
            if epochs and losses_tr and losses_val:
                plt.figure()
                plt.plot(epochs, losses_tr, label="Train Loss")
                plt.plot(epochs, losses_val, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{ds_name} – Loss Curve\nModel: {model_name} (Train vs Val)")
                plt.legend()
                fname = f"{ds_name}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {ds_name}: {e}")
            plt.close()

        # ---- 2) F1 curves ----
        try:
            if epochs and f1_tr and f1_val:
                plt.figure()
                plt.plot(epochs, f1_tr, label="Train F1")
                plt.plot(epochs, f1_val, label="Val F1")
                plt.xlabel("Epoch")
                plt.ylabel("Macro-F1")
                plt.title(f"{ds_name} – F1 Curve\nModel: {model_name} (Train vs Val)")
                plt.legend()
                fname = f"{ds_name}_f1_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating F1 curve for {ds_name}: {e}")
            plt.close()

        # ---- 3) confusion matrix ----
        try:
            if preds.size and gts.size and preds.shape == gts.shape:
                labels = np.unique(np.concatenate([preds, gts]))
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for p, t in zip(preds, gts):
                    cm[t, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xticks(range(len(labels)), labels, rotation=45)
                plt.yticks(range(len(labels)), labels)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{ds_name} – Confusion Matrix\nModel: {model_name}")
                fname = f"{ds_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname), bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()
