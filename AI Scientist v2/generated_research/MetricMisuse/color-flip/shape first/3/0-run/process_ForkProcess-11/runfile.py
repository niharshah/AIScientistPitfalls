import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
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
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    ds_name = "SPR_BENCH"
    rec = experiment_data[ds_name]

    # ------------------------------------------------- Plot 1: contrastive pretrain loss
    try:
        plt.figure()
        epochs = range(1, len(rec["losses"]["pretrain"]) + 1)
        plt.plot(epochs, rec["losses"]["pretrain"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Contrastive Loss")
        plt.title(f"{ds_name}: Contrastive Pre-training Loss")
        fn = os.path.join(working_dir, f"{ds_name}_pretrain_loss.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating pretrain loss plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 2: fine-tuning losses
    try:
        plt.figure()
        e2 = range(1, len(rec["losses"]["train"]) + 1)
        plt.plot(e2, rec["losses"]["train"], "--", label="train")
        plt.plot(e2, rec["losses"]["val"], "-", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title(f"{ds_name}: Train vs Val Loss")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_train_val_loss.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating train/val loss plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 3: validation accuracy & ACA
    try:
        plt.figure()
        plt.plot(e2, rec["metrics"]["val_acc"], label="Val Accuracy")
        plt.plot(e2, rec["metrics"]["val_aca"], label="Val ACA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_name}: Validation Metrics")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_val_metrics.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating validation metrics plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 4: test metric summary
    try:
        plt.figure()
        test_metrics = rec["test"]
        names = ["acc", "swa", "cwa", "aca"]
        scores = [test_metrics[n] for n in names]
        plt.bar(names, scores, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"{ds_name}: Test Metrics Summary\nLeft to Right: Acc, SWA, CWA, ACA")
        fn = os.path.join(working_dir, f"{ds_name}_test_metrics.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()
else:
    print("experiment_data not found or SPR_BENCH key missing.")
