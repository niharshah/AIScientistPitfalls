import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------
# Create/define working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------
# Load ALL experiment_data.npy files --------------------------------
experiment_data_path_list = [
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_bcfe02a5ab3f4dd684ef9a9122e90a08_proc_3013417/experiment_data.npy",
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_e2febd1765bd4f5380af51255da45c40_proc_3013418/experiment_data.npy",
    "experiments/2025-08-15_23-37-11_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_5ccc4b2f30f647a4a523758e44c32133_proc_3013420/experiment_data.npy",
]

all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        exp_full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        exp_dict = np.load(exp_full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading experiment data [{experiment_data_path}]: {e}")

# -------------------------------------------------------------------
# Aggregate by dataset ----------------------------------------------
aggregated = {}
for exp in all_experiment_data:
    for dname, ddata in exp.items():
        ds = aggregated.setdefault(
            dname,
            {
                "epochs": None,
                "train_losses": [],
                "val_losses": [],
                "val_metrics": [],
                "preds": [],
                "gts": [],
            },
        )
        # epochs (assume identical across runs; store once)
        if ds["epochs"] is None and ddata.get("epochs"):
            ds["epochs"] = np.array(ddata["epochs"])
        # store losses if available
        losses = ddata.get("losses", {})
        if "train" in losses:
            ds["train_losses"].append(np.array(losses["train"]))
        if "val" in losses:
            ds["val_losses"].append(np.array(losses["val"]))
        # store metrics
        metrics = ddata.get("metrics", {})
        if "val" in metrics:
            ds["val_metrics"].append(np.array(metrics["val"]))
        # preds / gts
        if "predictions" in ddata and "ground_truth" in ddata:
            ds["preds"].append(np.array(ddata["predictions"]))
            ds["gts"].append(np.array(ddata["ground_truth"]))


# -------------------------------------------------------------------
# Helper for mean and sem -------------------------------------------
def mean_sem(arr_list):
    """Stack list into array and return mean and SEM along axis 0."""
    arr = np.stack(arr_list, axis=0)
    mu = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mu, sem


# -------------------------------------------------------------------
# Iterate over datasets and create plots ----------------------------
for dname, ds in aggregated.items():
    epochs = ds["epochs"]
    # 1) Aggregated Train/Val loss
    try:
        if epochs is not None and ds["train_losses"] and ds["val_losses"]:
            train_mu, train_sem = mean_sem(ds["train_losses"])
            val_mu, val_sem = mean_sem(ds["val_losses"])

            plt.figure()
            plt.plot(epochs, train_mu, label="Train Loss – Mean", color="blue")
            plt.fill_between(
                epochs,
                train_mu - train_sem,
                train_mu + train_sem,
                color="blue",
                alpha=0.2,
                label="Train SEM",
            )
            plt.plot(epochs, val_mu, label="Val Loss – Mean", color="orange")
            plt.fill_between(
                epochs,
                val_mu - val_sem,
                val_mu + val_sem,
                color="orange",
                alpha=0.2,
                label="Val SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} Loss Curves (mean ± SEM)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregate_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dname}: {e}")
        plt.close()

    # 2) Aggregated Validation SCWA
    try:
        if epochs is not None and ds["val_metrics"]:
            val_mu, val_sem = mean_sem(ds["val_metrics"])
            plt.figure()
            plt.plot(epochs, val_mu, marker="o", color="green", label="Val SCWA – Mean")
            plt.fill_between(
                epochs,
                val_mu - val_sem,
                val_mu + val_sem,
                color="green",
                alpha=0.2,
                label="Val SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("SCWA")
            plt.title(f"{dname} Validation SCWA (mean ± SEM)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregate_val_SCWA.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated SCWA plot for {dname}: {e}")
        plt.close()

    # 3) Aggregated Confusion Matrix
    try:
        if ds["preds"] and ds["gts"]:
            preds_all = np.concatenate(ds["preds"])
            gts_all = np.concatenate(ds["gts"])
            num_classes = int(max(preds_all.max(), gts_all.max())) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts_all, preds_all):
                cm[t, p] += 1

            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                f"{dname} Confusion Matrix\nLeft: Ground Truth, Right: Aggregated Predictions"
            )
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
            fname = os.path.join(working_dir, f"{dname}_aggregate_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix for {dname}: {e}")
        plt.close()

    # ----------------------------------------------------------------
    # Print summary stats --------------------------------------------
    try:
        if ds["val_metrics"]:
            final_vals = [v[-1] for v in ds["val_metrics"]]
            print(
                f"{dname} Final Val SCWA: {np.mean(final_vals):.4f} ± {np.std(final_vals, ddof=1):.4f}"
            )
        if ds["preds"] and ds["gts"]:
            accs = []
            for pr, gt in zip(ds["preds"], ds["gts"]):
                accs.append((pr == gt).mean())
            print(
                f"{dname} Test Accuracy: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}"
            )
    except Exception as e:
        print(f"Error printing summary stats for {dname}: {e}")
