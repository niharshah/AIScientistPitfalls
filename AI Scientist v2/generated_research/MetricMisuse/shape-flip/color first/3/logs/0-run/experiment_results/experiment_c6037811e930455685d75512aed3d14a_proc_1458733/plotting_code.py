import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------------
# (1) Per-dataset learning curves & losses
for ds_name, ds_dict in experiment_data.items():
    # ---------------- BWA curve ----------------
    try:
        epochs = np.arange(1, len(ds_dict["metrics"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, ds_dict["metrics"]["train"], label="Train BWA")
        plt.plot(epochs, ds_dict["metrics"]["val"], label="Validation BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"{ds_name}: Train vs Val BWA")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds_name.lower()}_bwa_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting BWA for {ds_name}: {e}")
        plt.close()

    # ---------------- StrWA curve ----------------
    try:
        if "StrWA" in ds_dict and ds_dict["StrWA"]["train"]:
            epochs = np.arange(1, len(ds_dict["StrWA"]["train"]) + 1)
            plt.figure()
            plt.plot(epochs, ds_dict["StrWA"]["train"], "--", label="Train StrWA")
            plt.plot(epochs, ds_dict["StrWA"]["val"], "--", label="Validation StrWA")
            plt.xlabel("Epoch")
            plt.ylabel("StrWA")
            plt.title(f"{ds_name}: Train vs Val StrWA")
            plt.legend()
            plt.tight_layout()
            fname = f"{ds_name.lower()}_strwa_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error plotting StrWA for {ds_name}: {e}")
        plt.close()

    # ---------------- Loss curve ----------------
    try:
        epochs = np.arange(1, len(ds_dict["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, ds_dict["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ds_dict["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("CE Loss")
        plt.title(f"{ds_name}: Train vs Val Loss")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds_name.lower()}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {ds_name}: {e}")
        plt.close()

# ---------------------------------------------------------------------
# (2) Bar chart comparing test BWA across datasets
try:
    ds_names, test_bwas = [], []
    for ds_name, ds_dict in experiment_data.items():
        bw = ds_dict.get("test_metrics", {}).get("BWA", None)
        if bw is not None:
            ds_names.append(ds_name)
            test_bwas.append(bw)
    if ds_names:
        plt.figure()
        xpos = np.arange(len(ds_names))
        plt.bar(xpos, test_bwas, color="skyblue")
        plt.xticks(xpos, ds_names, rotation=45, ha="right")
        plt.ylabel("Test BWA")
        plt.title("Test BWA Comparison Across Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "test_bwa_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error creating BWA comparison bar chart: {e}")
    plt.close()

# ---------------------------------------------------------------------
# (3) Confusion matrices (≤5 datasets)
for i, (ds_name, ds_dict) in enumerate(experiment_data.items()):
    if i >= 5:
        break
    try:
        preds = np.array(ds_dict.get("predictions", []))
        gts = np.array(ds_dict.get("ground_truth", []))
        if preds.size == 0 or gts.size == 0:
            continue
        num_classes = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_name}: Confusion Matrix")
        for (r, c), v in np.ndenumerate(cm):
            plt.text(c, r, str(v), ha="center", va="center", fontsize=7)
        plt.tight_layout()
        fname = f"{ds_name.lower()}_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()
