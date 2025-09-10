import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch nested keys
def get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


for run_key, datasets in experiment_data.items():
    for ds_name, ds_data in datasets.items():
        epochs = np.array(get(ds_data, "epochs", default=[]))
        if epochs.size == 0:
            continue  # nothing to plot

        # Gather curves
        loss_tr = np.array(get(ds_data, "losses", "train", default=[]))
        loss_val = np.array(get(ds_data, "losses", "val", default=[]))

        def extract_metric(split, field):
            lst = get(ds_data, "metrics", split, default=[])
            return np.array([m.get(field) for m in lst]) if lst else np.array([])

        acc_tr, acc_val = extract_metric("train", "acc"), extract_metric("val", "acc")
        mcc_tr, mcc_val = extract_metric("train", "MCC"), extract_metric("val", "MCC")
        rma_tr, rma_val = extract_metric("train", "RMA"), extract_metric("val", "RMA")

        # 1) Loss curve
        try:
            plt.figure()
            plt.plot(epochs, loss_tr, label="Train")
            plt.plot(epochs, loss_val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"{ds_name} Loss Curve (run: {run_key})\nTrain vs Validation")
            fn = os.path.join(working_dir, f"{ds_name}_loss_curve.png")
            plt.savefig(fn)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name}: {e}")
            plt.close()

        # 2) Accuracy curve
        try:
            plt.figure()
            plt.plot(epochs, acc_tr, label="Train")
            plt.plot(epochs, acc_val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title(f"{ds_name} Accuracy Curve (run: {run_key})\nTrain vs Validation")
            fn = os.path.join(working_dir, f"{ds_name}_accuracy_curve.png")
            plt.savefig(fn)
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {ds_name}: {e}")
            plt.close()

        # 3) MCC curve
        try:
            plt.figure()
            plt.plot(epochs, mcc_tr, label="Train")
            plt.plot(epochs, mcc_val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.legend()
            plt.title(f"{ds_name} MCC Curve (run: {run_key})\nTrain vs Validation")
            fn = os.path.join(working_dir, f"{ds_name}_mcc_curve.png")
            plt.savefig(fn)
            plt.close()
        except Exception as e:
            print(f"Error creating MCC plot for {ds_name}: {e}")
            plt.close()

        # 4) RMA curve
        try:
            plt.figure()
            plt.plot(epochs, rma_tr, label="Train")
            plt.plot(epochs, rma_val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Rule Macro Acc.")
            plt.legend()
            plt.title(f"{ds_name} RMA Curve (run: {run_key})\nTrain vs Validation")
            fn = os.path.join(working_dir, f"{ds_name}_rma_curve.png")
            plt.savefig(fn)
            plt.close()
        except Exception as e:
            print(f"Error creating RMA plot for {ds_name}: {e}")
            plt.close()

        # 5) Confusion matrix on test set
        try:
            preds = np.array(get(ds_data, "predictions", default=[]))
            gts = np.array(get(ds_data, "ground_truth", default=[]))
            if preds.size and gts.size and preds.size == gts.size:
                cm = np.zeros((2, 2), dtype=int)
                for p, g in zip(preds, gts):
                    cm[int(g), int(p)] += 1
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        plt.text(
                            j,
                            i,
                            str(cm[i, j]),
                            ha="center",
                            va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                        )
                plt.title(
                    f"{ds_name} Confusion Matrix (run: {run_key})\nLeft: Ground Truth, Right: Predictions"
                )
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.colorbar()
                fn = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
                plt.savefig(fn)
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()

        # Print final test metrics
        tm = get(ds_data, "test_metrics", default={})
        if tm:
            print(
                f"{ds_name} final test metrics → loss:{tm.get('loss'):.4f}, "
                f"acc:{tm.get('acc'):.3f}, MCC:{tm.get('MCC'):.3f}, RMA:{tm.get('RMA'):.3f}"
            )
