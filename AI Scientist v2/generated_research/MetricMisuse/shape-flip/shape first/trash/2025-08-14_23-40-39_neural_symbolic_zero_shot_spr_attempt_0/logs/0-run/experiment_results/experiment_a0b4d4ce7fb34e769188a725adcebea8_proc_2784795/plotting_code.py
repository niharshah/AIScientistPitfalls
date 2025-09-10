import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

fig_cnt = 0
MAX_FIGS = 5

for model_name, datasets in experiment_data.items():
    for dataset_name, data in datasets.items():
        metrics = data.get("metrics", {})
        # ------------- Loss curves ---------------------------------
        try:
            if fig_cnt < MAX_FIGS:
                plt.figure()
                epochs = np.arange(1, len(metrics.get("train_loss", [])) + 1)
                plt.plot(epochs, metrics.get("train_loss", []), label="Train Loss")
                plt.plot(epochs, metrics.get("val_loss", []), label="Val Loss")
                plt.title(f"{dataset_name} Loss Curves\nTraining vs Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                fname = f"{dataset_name}_train_val_loss.png"
                plt.savefig(os.path.join(working_dir, fname))
                fig_cnt += 1
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # ------------- SWA curve -----------------------------------
        try:
            if fig_cnt < MAX_FIGS:
                plt.figure()
                plt.plot(epochs, metrics.get("val_swa", []), marker="o")
                plt.title(f"{dataset_name} Validation Shape-Weighted Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("SWA")
                fname = f"{dataset_name}_val_SWA.png"
                plt.savefig(os.path.join(working_dir, fname))
                fig_cnt += 1
            plt.close()
        except Exception as e:
            print(f"Error creating SWA plot: {e}")
            plt.close()

        # Helper to build confusion matrices
        def plot_confusion(split):
            preds = np.array(data["predictions"].get(split, []))
            gts = np.array(data["ground_truth"].get(split, []))
            if preds.size == 0 or gts.size == 0:
                return
            n_cls = int(max(preds.max(initial=0), gts.max(initial=0))) + 1
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(f"{dataset_name} {split.capitalize()} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(n_cls))
            plt.yticks(range(n_cls))

        # ------------- Dev confusion matrix ------------------------
        try:
            if fig_cnt < MAX_FIGS:
                plt.figure()
                plot_confusion("dev")
                fname = f"{dataset_name}_dev_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                fig_cnt += 1
            plt.close()
        except Exception as e:
            print(f"Error creating dev confusion matrix: {e}")
            plt.close()

        # ------------- Test confusion matrix -----------------------
        try:
            if fig_cnt < MAX_FIGS:
                plt.figure()
                plot_confusion("test")
                fname = f"{dataset_name}_test_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                fig_cnt += 1
            plt.close()
        except Exception as e:
            print(f"Error creating test confusion matrix: {e}")
            plt.close()
