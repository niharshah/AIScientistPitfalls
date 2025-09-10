import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
# prepare working directory                                          #
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# 1. Load ALL experiment_data dicts                                  #
# ------------------------------------------------------------------ #
try:
    experiment_data_path_list = [
        "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_8341316809804c6ab45e315e7b5297e8_proc_1599549/experiment_data.npy",
        "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_629b199eb09e410fbc595e4b688dd54f_proc_1599547/experiment_data.npy",
        "experiments/2025-08-31_02-26-58_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_70b0c99b230d4abab9200bccaef228cb_proc_1599550/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

if not all_experiment_data:
    exit()

# identify datasets present in the first run
dataset_names = list(all_experiment_data[0].keys())


# ------------------------------------------------------------------ #
# helper to compute mean & sem over runs                             #
# ------------------------------------------------------------------ #
def aggregate_epoch_pairs(list_of_epoch_pairs):
    """
    list_of_epoch_pairs: list of list[(epoch,value)]
    Returns: epochs (np.array), mean (np.array), sem (np.array)
    Assumes every run contains the same epoch list; if not, keeps common epochs
    """
    # build set of common epochs
    epoch_sets = [set(ep for ep, _ in run) for run in list_of_epoch_pairs]
    common_epochs = sorted(set.intersection(*epoch_sets))
    if not common_epochs:
        return None, None, None
    values = []
    for run in list_of_epoch_pairs:
        run_dict = dict(run)
        values.append([run_dict[e] for e in common_epochs])
    values = np.array(values, dtype=float)  # shape (n_runs, n_epochs)
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return np.array(common_epochs), mean, sem


# ------------------------------------------------------------------ #
# 2. Loop over datasets and create aggregated plots                  #
# ------------------------------------------------------------------ #
for ds in dataset_names:
    # gather data across runs; if a run is missing the dataset, ignore it
    runs_with_ds = [r for r in all_experiment_data if ds in r]
    if len(runs_with_ds) < 2:  # need at least 2 to show sem
        continue

    # -------------------------------------------------------------- #
    # 2a. Train & Val Loss                                           #
    # -------------------------------------------------------------- #
    try:
        train_epoch_pairs = [run[ds]["losses"]["train"] for run in runs_with_ds]
        val_epoch_pairs = [run[ds]["losses"]["val"] for run in runs_with_ds]

        tr_epochs, tr_mean, tr_sem = aggregate_epoch_pairs(train_epoch_pairs)
        val_epochs, val_mean, val_sem = aggregate_epoch_pairs(val_epoch_pairs)

        if tr_epochs is None or val_epochs is None:
            raise ValueError("Epoch alignment failed.")

        plt.figure()
        plt.plot(tr_epochs, tr_mean, label="Mean Train Loss", color="tab:blue")
        plt.fill_between(
            tr_epochs,
            tr_mean - tr_sem,
            tr_mean + tr_sem,
            alpha=0.3,
            color="tab:blue",
            label="Train SEM",
        )

        plt.plot(val_epochs, val_mean, label="Mean Val Loss", color="tab:orange")
        plt.fill_between(
            val_epochs,
            val_mean - val_sem,
            val_mean + val_sem,
            alpha=0.3,
            color="tab:orange",
            label="Val SEM",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title(
            f"{ds}: Aggregated Train vs Validation Loss\n(Mean ± SEM over {len(runs_with_ds)} runs)"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_aggregated_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds}: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # 2b. Validation Metrics (CWA, SWA, HCSA)                        #
    # -------------------------------------------------------------- #
    try:
        metric_epoch_pairs = [
            runs_with_ds[i][ds]["metrics"]["val"] for i in range(len(runs_with_ds))
        ]
        # split tuples into separate lists per metric
        cwa_pairs, swa_pairs, hcs_pairs = [], [], []
        for run in metric_epoch_pairs:
            epochs, cwa, swa, hcs = zip(*run)
            cwa_pairs.append(list(zip(epochs, cwa)))
            swa_pairs.append(list(zip(epochs, swa)))
            hcs_pairs.append(list(zip(epochs, hcs)))

        ep_cwa, mean_cwa, sem_cwa = aggregate_epoch_pairs(cwa_pairs)
        ep_swa, mean_swa, sem_swa = aggregate_epoch_pairs(swa_pairs)
        ep_hcs, mean_hcs, sem_hcs = aggregate_epoch_pairs(hcs_pairs)

        if ep_cwa is None:
            raise ValueError("Epoch alignment failed for metrics.")

        plt.figure()
        plt.plot(ep_cwa, mean_cwa, label="Mean CWA", color="tab:green")
        plt.fill_between(
            ep_cwa,
            mean_cwa - sem_cwa,
            mean_cwa + sem_cwa,
            alpha=0.3,
            color="tab:green",
            label="CWA SEM",
        )

        plt.plot(ep_swa, mean_swa, label="Mean SWA", color="tab:red")
        plt.fill_between(
            ep_swa,
            mean_swa - sem_swa,
            mean_swa + sem_swa,
            alpha=0.3,
            color="tab:red",
            label="SWA SEM",
        )

        plt.plot(ep_hcs, mean_hcs, label="Mean HCSA", color="tab:purple")
        plt.fill_between(
            ep_hcs,
            mean_hcs - sem_hcs,
            mean_hcs + sem_hcs,
            alpha=0.3,
            color="tab:purple",
            label="HCSA SEM",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(
            f"{ds}: Aggregated Validation Metrics\n(Mean ± SEM over {len(runs_with_ds)} runs)"
        )
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_aggregated_val_metrics.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated metric plot for {ds}: {e}")
        plt.close()
