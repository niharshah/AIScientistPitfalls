import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# list of experiment_data.npy files provided by the platform
experiment_data_path_list = [
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_7e4fab683bf74682b8e742637db71e2a_proc_3154416/experiment_data.npy",
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_9c2a530ad2db4daf92bdf17284100047_proc_3154415/experiment_data.npy",
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_9b1a05b08e0446ffb4fcb9080661bee3_proc_3154417/experiment_data.npy",
]

# ------------------------------------------------------------------ #
# load all experiment dicts
all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_dict = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_dict)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ------------------------------------------------------------------ #
# regroup by dataset name
datasets = {}
for exp in all_experiment_data:
    for dset_name, dset_dict in exp.items():
        datasets.setdefault(dset_name, []).append(dset_dict)


# ------------------------------------------------------------------ #
def stack_and_truncate(list_of_arrays):
    """Stack 1-D arrays (possibly different length) by truncating to min length."""
    min_len = min([len(a) for a in list_of_arrays])
    arr = np.stack([a[:min_len] for a in list_of_arrays], axis=0)
    return arr


for dset_name, runs in datasets.items():
    # ---------- collect per-epoch metrics ---------- #
    train_losses = stack_and_truncate(
        [np.array(r["losses"]["train_loss"]) for r in runs]
    )
    val_losses = stack_and_truncate([np.array(r["losses"]["val_loss"]) for r in runs])
    train_accs = stack_and_truncate([np.array(r["metrics"]["train_acc"]) for r in runs])
    val_accs = stack_and_truncate([np.array(r["metrics"]["val_acc"]) for r in runs])
    epochs = np.arange(1, train_losses.shape[1] + 1)

    # ---------- aggregated Loss Curves ------------- #
    try:
        plt.figure()
        # mean ± sem
        for arr, lbl, color in [
            (train_losses, "Train", "tab:blue"),
            (val_losses, "Validation", "tab:orange"),
        ]:
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            plt.plot(epochs, mean, label=f"{lbl} Mean", color=color)
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color=color,
                label=f"{lbl} ±1 SEM",
            )
        plt.title(f"{dset_name} Loss Curves (Mean ± SEM)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"{dset_name.lower()}_aggregated_loss_curves.png".replace(" ", "_")
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset_name}: {e}")
        plt.close()

    # ---------- aggregated Accuracy Curves --------- #
    try:
        plt.figure()
        for arr, lbl, color in [
            (train_accs, "Train", "tab:green"),
            (val_accs, "Validation", "tab:red"),
        ]:
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            plt.plot(epochs, mean, label=f"{lbl} Mean", color=color)
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color=color,
                label=f"{lbl} ±1 SEM",
            )
        plt.title(f"{dset_name} Accuracy Curves (Mean ± SEM)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"{dset_name.lower()}_aggregated_accuracy_curves.png".replace(" ", "_")
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {dset_name}: {e}")
        plt.close()

    # ---------- test accuracy per run -------------- #
    try:
        test_accs = []
        for r in runs:
            preds = np.array(r["predictions"])
            gts = np.array(r["ground_truth"])
            test_accs.append((preds == gts).mean())
        test_accs = np.array(test_accs)
        mean_acc = test_accs.mean()
        sem_acc = test_accs.std(ddof=1) / np.sqrt(len(test_accs))
        plt.figure()
        plt.bar(
            np.arange(len(test_accs)),
            test_accs,
            color="skyblue",
            label="Individual Runs",
        )
        plt.errorbar(
            x=len(test_accs) + 0.5,
            y=mean_acc,
            yerr=sem_acc,
            fmt="o",
            color="red",
            label=f"Mean ± SEM ({mean_acc:.3f}±{sem_acc:.3f})",
        )
        plt.xticks(
            list(range(len(test_accs))) + [len(test_accs) + 0.5],
            [f"Run {i}" for i in range(len(test_accs))] + ["Mean"],
        )
        plt.ylim(0, 1)
        plt.ylabel("Test Accuracy")
        plt.title(f"{dset_name} Test Accuracy Across Runs")
        plt.legend()
        fname = f"{dset_name.lower()}_test_accuracy_bar.png".replace(" ", "_")
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print(f"{dset_name}: mean test accuracy = {mean_acc:.4f} ± {sem_acc:.4f} (SEM)")
    except Exception as e:
        print(f"Error creating test accuracy bar plot for {dset_name}: {e}")
        plt.close()
