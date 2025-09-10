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
# per-dataset visualisations
for ds_name, ds_dict in experiment_data.items():
    # --------------- 1) loss curve -----------------------------------
    try:
        epochs = np.arange(1, len(ds_dict["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, ds_dict["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ds_dict["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name}: Train vs Val Loss")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds_name}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
    finally:
        plt.close()

    # --------------- 2) BWA curve ------------------------------------
    try:
        train_bwa = [m["BWA"] for m in ds_dict["metrics"]["train"]]
        val_bwa = [m["BWA"] for m in ds_dict["metrics"]["val"]]
        epochs = np.arange(1, len(train_bwa) + 1)
        plt.figure()
        plt.plot(epochs, train_bwa, label="Train BWA")
        plt.plot(epochs, val_bwa, label="Val BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"{ds_name}: Train vs Val BWA")
        plt.legend()
        plt.tight_layout()
        fname = f"{ds_name}_bwa_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating BWA curve for {ds_name}: {e}")
    finally:
        plt.close()

    # --------------- 3) test metric bars -----------------------------
    try:
        tm = ds_dict.get("test_metrics", {})
        metrics = ["BWA", "CWA", "SWA", "StrWA"]
        vals = [tm.get(m, np.nan) for m in metrics]
        plt.figure()
        plt.bar(metrics, vals, color="skyblue")
        for i, v in enumerate(vals):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.ylabel("Score")
        plt.title(f"{ds_name}: Test Metrics")
        plt.tight_layout()
        fname = f"{ds_name}_test_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating test-metric bar chart for {ds_name}: {e}")
    finally:
        plt.close()

    # --------------- 4) confusion matrix -----------------------------
    try:
        preds = np.array(ds_dict["predictions"])
        gts = np.array(ds_dict["ground_truth"])
        num_classes = int(max(preds.max(), gts.max()) + 1)
        conf = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            conf[gt, pr] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(conf, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_name}: Confusion Matrix")
        for (i, j), v in np.ndenumerate(conf):
            plt.text(j, i, str(v), ha="center", va="center", fontsize=7)
        plt.tight_layout()
        fname = f"{ds_name}_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
    finally:
        plt.close()

    # --------------- 5) print test metrics ---------------------------
    if "test_metrics" in ds_dict:
        print(f"{ds_name} TEST -> ", ds_dict["test_metrics"])

# ---------------------------------------------------------------------
# cross-dataset BWA comparison (only if >1 datasets)
try:
    if len(experiment_data) > 1:
        dsn, bwa_vals = [], []
        for k, v in experiment_data.items():
            if "test_metrics" in v and "BWA" in v["test_metrics"]:
                dsn.append(k)
                bwa_vals.append(v["test_metrics"]["BWA"])
        if dsn:
            plt.figure()
            x = np.arange(len(dsn))
            plt.bar(x, bwa_vals, color="salmon")
            plt.xticks(x, dsn, rotation=45, ha="right")
            plt.ylabel("Test BWA")
            plt.title("Dataset Comparison: Test BWA")
            for i, v in enumerate(bwa_vals):
                plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            fname = "cross_dataset_test_bwa.png"
            plt.savefig(os.path.join(working_dir, fname))
            print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating cross-dataset comparison: {e}")
finally:
    plt.close()
