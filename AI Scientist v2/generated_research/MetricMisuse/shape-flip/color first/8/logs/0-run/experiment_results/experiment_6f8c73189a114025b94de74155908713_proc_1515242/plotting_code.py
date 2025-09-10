import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

final_cxa = {}

# ---------- iterate over datasets ----------
for ds_name, ds_content in experiment_data.items():
    losses = ds_content.get("losses", {})
    metrics = ds_content.get("metrics", {})
    preds = np.array(ds_content.get("predictions", []))
    gts = np.array(ds_content.get("ground_truth", []))

    # --------- plot 1: loss curves ---------
    try:
        plt.figure()
        if losses.get("train"):
            plt.plot(losses["train"], label="Train")
        if losses.get("val"):
            plt.plot(losses["val"], label="Validation")
        plt.title(f"{ds_name} Cross-Entropy Loss\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"{ds_name}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # --------- helper: extract metric array ---------
    def m_arr(split, key):
        return [d[key] for d in metrics.get(split, [])]

    # --------- plot 2: accuracy curves ---------
    try:
        plt.figure()
        if metrics.get("train"):
            plt.plot(m_arr("train", "acc"), label="Train")
        if metrics.get("val"):
            plt.plot(m_arr("val", "acc"), label="Validation")
        plt.title(f"{ds_name} Accuracy\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"{ds_name}_accuracy_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {ds_name}: {e}")
        plt.close()

    # --------- plot 3: CXA curves ---------
    try:
        plt.figure()
        if metrics.get("train"):
            plt.plot(m_arr("train", "cxa"), label="Train")
        if metrics.get("val"):
            plt.plot(m_arr("val", "cxa"), label="Validation")
        plt.title(
            f"{ds_name} Complexity-Weighted Accuracy\nLeft: Train, Right: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("CXA")
        plt.legend()
        fname = f"{ds_name}_cxa_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CXA curve for {ds_name}: {e}")
        plt.close()

    # --------- plot 4: confusion matrix ---------
    try:
        if preds.size and gts.size:
            num_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for g, p in zip(gts, preds):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{ds_name} Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            ticks = np.arange(num_classes)
            plt.xticks(ticks, [f"c{i}" for i in ticks])
            plt.yticks(ticks, [f"c{i}" for i in ticks])
            plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
            plt.close()
        else:
            print(
                f"Skipping confusion matrix for {ds_name}: empty predictions or ground truth."
            )
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # --------- print summary metrics ----------
    if metrics.get("val"):
        final_acc = m_arr("val", "acc")[-1]
        final_cxa_val = m_arr("val", "cxa")[-1]
        final_cxa[ds_name] = final_cxa_val
        print(
            f"{ds_name} final validation accuracy: {final_acc:.4f}, CXA: {final_cxa_val:.4f}"
        )

# ---------- comparison plot across datasets ----------
if len(final_cxa) > 1:
    try:
        plt.figure()
        names, cxa_vals = zip(*final_cxa.items())
        plt.bar(names, cxa_vals, color="steelblue")
        plt.title("Final Validation CXA Comparison\nDataset-wise Performance")
        plt.ylabel("CXA")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "dataset_cxa_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating dataset comparison plot: {e}")
        plt.close()
