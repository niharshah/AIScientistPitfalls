import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# ---------------------------------------------------------------------
# basic setup
# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load all experiment_data files
# ---------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_f4bf07e193614b3fa4e9ef30e942f9c6_proc_1726493/experiment_data.npy",
        "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_e19aadfebe0444b1b6d1f82c75b5ac29_proc_1726491/experiment_data.npy",
        "experiments/2025-08-31_14-12-02_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_4a86d7c92b4f48d19f539c3a1e3e8ccd_proc_1726492/experiment_data.npy",
    ]

    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ---------------------------------------------------------------------
# aggregate helper: defaultdict(dict(epoch -> list))
# ---------------------------------------------------------------------
agg = defaultdict(
    lambda: {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "cshm": defaultdict(list),
    }
)

for run in all_experiment_data:
    for exp_key, exp_dict in run.get("num_epochs", {}).items():
        # training losses
        for epoch, loss in exp_dict["losses"]["train"]:
            agg[exp_key]["train"][epoch].append(loss)
        # validation losses
        for epoch, loss in exp_dict["losses"]["val"]:
            agg[exp_key]["val"][epoch].append(loss)
        # validation metrics: (epoch, cwa, swa, cshm)
        for epoch, _, _, cshm in exp_dict["metrics"]["val"]:
            agg[exp_key]["cshm"][epoch].append(cshm)

# ---------------------------------------------------------------------
# Plot 1: mean ± SEM training/validation loss
# ---------------------------------------------------------------------
try:
    if not agg:
        raise ValueError("No aggregated data available for plotting loss curves.")

    plt.figure()
    for exp_key, d in agg.items():
        # -------- train --------
        epochs = sorted(d["train"].keys())
        train_means = [np.mean(d["train"][ep]) for ep in epochs]
        train_sems = [
            (
                np.std(d["train"][ep], ddof=1) / np.sqrt(len(d["train"][ep]))
                if len(d["train"][ep]) > 1
                else 0.0
            )
            for ep in epochs
        ]
        plt.plot(epochs, train_means, "--", label=f"train mean-{exp_key}")
        plt.fill_between(
            epochs,
            np.array(train_means) - np.array(train_sems),
            np.array(train_means) + np.array(train_sems),
            alpha=0.2,
        )

        # -------- val --------
        epochs_val = sorted(d["val"].keys())
        val_means = [np.mean(d["val"][ep]) for ep in epochs_val]
        val_sems = [
            (
                np.std(d["val"][ep], ddof=1) / np.sqrt(len(d["val"][ep]))
                if len(d["val"][ep]) > 1
                else 0.0
            )
            for ep in epochs_val
        ]
        plt.plot(epochs_val, val_means, "-", label=f"val mean-{exp_key}")
        plt.fill_between(
            epochs_val,
            np.array(val_means) - np.array(val_sems),
            np.array(val_means) + np.array(val_sems),
            alpha=0.2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Synthetic SPR: Mean Training and Validation Loss (±SEM)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_mean_loss_curves_with_sem.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating mean loss curve plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# Plot 2: mean ± SEM validation CSHM
# ---------------------------------------------------------------------
try:
    if not agg:
        raise ValueError("No aggregated data available for plotting CSHM curves.")

    plt.figure()
    for exp_key, d in agg.items():
        epochs = sorted(d["cshm"].keys())
        cshm_means = [np.mean(d["cshm"][ep]) for ep in epochs]
        cshm_sems = [
            (
                np.std(d["cshm"][ep], ddof=1) / np.sqrt(len(d["cshm"][ep]))
                if len(d["cshm"][ep]) > 1
                else 0.0
            )
            for ep in epochs
        ]
        plt.plot(epochs, cshm_means, label=f"CSHM mean-{exp_key}")
        plt.fill_between(
            epochs,
            np.array(cshm_means) - np.array(cshm_sems),
            np.array(cshm_means) + np.array(cshm_sems),
            alpha=0.2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("CSHM")
    plt.title("Synthetic SPR: Mean Validation Colour-Shape Harmonic Mean (±SEM)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_mean_validation_cshm_with_sem.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating mean CSHM plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# Print out final aggregated statistics for quick inspection
# ---------------------------------------------------------------------
print("Aggregated mean (±SEM) at final epoch per setting:")
for exp_key, d in agg.items():
    if d["val"]:
        last_epoch = max(d["val"].keys())
        mean_loss = np.mean(d["val"][last_epoch])
        sem_loss = (
            np.std(d["val"][last_epoch], ddof=1) / np.sqrt(len(d["val"][last_epoch]))
            if len(d["val"][last_epoch]) > 1
            else 0.0
        )
        print(
            f"{exp_key} | epoch {last_epoch}: val loss = {mean_loss:.4f} ± {sem_loss:.4f}"
        )
    if d["cshm"]:
        last_epoch = max(d["cshm"].keys())
        mean_cshm = np.mean(d["cshm"][last_epoch])
        sem_cshm = (
            np.std(d["cshm"][last_epoch], ddof=1) / np.sqrt(len(d["cshm"][last_epoch]))
            if len(d["cshm"][last_epoch]) > 1
            else 0.0
        )
        print(
            f"{exp_key} | epoch {last_epoch}: val CSHM = {mean_cshm:.4f} ± {sem_cshm:.4f}"
        )
