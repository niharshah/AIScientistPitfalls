import matplotlib.pyplot as plt
import numpy as np
import os

# -------- mandatory working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -------- helper for safe plotting --------
def safe_save(fig, fname):
    path = os.path.join(working_dir, fname)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# -------- iterate over datasets -----------
for ds_name, ds_dict in experiment_data.items():
    # ----- 1) BWA learning curve ----------
    try:
        epochs = np.arange(1, len(ds_dict["metrics"]["train"]) + 1)
        fig = plt.figure()
        plt.plot(epochs, ds_dict["metrics"]["train"], label="Train BWA")
        plt.plot(epochs, ds_dict["metrics"]["val"], label="Validation BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"{ds_name.upper()} – BWA Learning Curve")
        plt.legend()
        safe_save(fig, f"{ds_name}_bwa_curve.png")
    except Exception as e:
        print(f"Error creating BWA curve for {ds_name}: {e}")
        plt.close()

    # ----- 2) Loss learning curve ---------
    try:
        epochs = np.arange(1, len(ds_dict["losses"]["train"]) + 1)
        fig = plt.figure()
        plt.plot(epochs, ds_dict["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ds_dict["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name.upper()} – Loss Learning Curve")
        plt.legend()
        safe_save(fig, f"{ds_name}_loss_curve.png")
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # ----- 3) Test metric bar chart -------
    try:
        tmet = ds_dict.get("test_metrics", {})
        labels = ["BWA", "CWA", "SWA", "StrWA"]
        values = [tmet.get(k, np.nan) for k in labels]
        fig = plt.figure()
        plt.bar(labels, values, color="skyblue")
        plt.ylabel("Score")
        plt.title(f"{ds_name.upper()} – Test Metrics")
        safe_save(fig, f"{ds_name}_test_metrics.png")
    except Exception as e:
        print(f"Error creating test metric bar chart for {ds_name}: {e}")
        plt.close()

    # ----- 4) Confusion matrix ------------
    try:
        preds = np.array(ds_dict["predictions"])
        gts = np.array(ds_dict["ground_truth"])
        num_c = int(max(preds.max(), gts.max()) + 1)
        conf = np.zeros((num_c, num_c), dtype=int)
        for gt, pr in zip(gts, preds):
            conf[gt, pr] += 1
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(conf, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_name.upper()} – Confusion Matrix")
        for (i, j), v in np.ndenumerate(conf):
            plt.text(j, i, str(v), ha="center", va="center", fontsize=7)
        safe_save(fig, f"{ds_name}_confusion_matrix.png")
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # ----- 5) Print numeric results -------
    if "test_metrics" in ds_dict:
        print(f"{ds_name.upper()} – Test metrics:", ds_dict["test_metrics"])
