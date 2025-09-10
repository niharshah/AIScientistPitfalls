import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) Load every experiment_data.npy that the task description listed
# ------------------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_2e183cbfb5934f268deb295e86456cff_proc_3341511/experiment_data.npy",
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_2461cde302234e38bd0533751f0b7dd4_proc_3341509/experiment_data.npy",
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_f6e098d882534cf282009f6bbe423caa_proc_3341512/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        exp_d = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(exp_d)
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No experiment files could be loaded – exiting.")
    exit()

# ------------------------------------------------------------------
# 2) Aggregate and plot for every optimisation strategy / dataset
# ------------------------------------------------------------------
for strategy in all_experiment_data[0].keys():
    # Determine every dataset that appears under this strategy
    datasets = set()
    for exp in all_experiment_data:
        datasets.update(exp[strategy].keys())

    for dset in datasets:
        # Gather per-run sequences ------------------------------------------------
        tr_loss_runs, va_loss_runs = [], []
        tr_mcc_runs, va_mcc_runs = [], []

        for exp in all_experiment_data:
            if strategy not in exp or dset not in exp[strategy]:
                continue
            exp_block = exp[strategy][dset]

            tr_loss, va_loss = exp_block["losses"]["train"], exp_block["losses"]["val"]
            tr_mcc, va_mcc = exp_block["metrics"]["train"], exp_block["metrics"]["val"]
            cfgs = exp_block["configs"]

            ptr = 0
            for cfg in cfgs:
                ep = cfg["epochs"]
                tr_loss_runs.append(tr_loss[ptr : ptr + ep])
                va_loss_runs.append(va_loss[ptr : ptr + ep])
                tr_mcc_runs.append(tr_mcc[ptr : ptr + ep])
                va_mcc_runs.append(va_mcc[ptr : ptr + ep])
                ptr += ep

        n_runs = len(tr_loss_runs)
        if n_runs == 0:
            print(f"No runs found for {strategy}/{dset}")
            continue

        # Pad to equal length with NaNs ------------------------------------------
        def pad_sequences(seq_list):
            max_len = max(len(s) for s in seq_list)
            mat = np.full((len(seq_list), max_len), np.nan, dtype=float)
            for i, s in enumerate(seq_list):
                mat[i, : len(s)] = s
            return mat

        tr_loss_mat, va_loss_mat = pad_sequences(tr_loss_runs), pad_sequences(
            va_loss_runs
        )
        tr_mcc_mat, va_mcc_mat = pad_sequences(tr_mcc_runs), pad_sequences(va_mcc_runs)

        # Mean & Standard Error (nan-aware) --------------------------------------
        def mean_and_sem(mat):
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(
                np.sum(~np.isnan(mat), axis=0)
            )
            return mean, sem

        tr_loss_mean, tr_loss_sem = mean_and_sem(tr_loss_mat)
        va_loss_mean, va_loss_sem = mean_and_sem(va_loss_mat)
        tr_mcc_mean, tr_mcc_sem = mean_and_sem(tr_mcc_mat)
        va_mcc_mean, va_mcc_sem = mean_and_sem(va_mcc_mat)

        epochs_axis = np.arange(1, len(tr_loss_mean) + 1)

        # ----------------------------------------------------------------------
        # 3) Plot aggregated curves with SEM bands
        # ----------------------------------------------------------------------
        try:
            fig, ax = plt.subplots(2, 1, figsize=(7, 10))

            # -------- Loss subplot ----------
            ax[0].plot(epochs_axis, tr_loss_mean, label="Train Mean", color="tab:blue")
            ax[0].fill_between(
                epochs_axis,
                tr_loss_mean - tr_loss_sem,
                tr_loss_mean + tr_loss_sem,
                color="tab:blue",
                alpha=0.3,
                label="Train ± SEM",
            )
            ax[0].plot(epochs_axis, va_loss_mean, label="Val Mean", color="tab:orange")
            ax[0].fill_between(
                epochs_axis,
                va_loss_mean - va_loss_sem,
                va_loss_mean + va_loss_sem,
                color="tab:orange",
                alpha=0.3,
                label="Val ± SEM",
            )
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("BCE Loss")
            ax[0].set_title("Aggregated BCE Loss")
            ax[0].legend()

            # -------- MCC subplot -----------
            ax[1].plot(epochs_axis, tr_mcc_mean, label="Train Mean", color="tab:green")
            ax[1].fill_between(
                epochs_axis,
                tr_mcc_mean - tr_mcc_sem,
                tr_mcc_mean + tr_mcc_sem,
                color="tab:green",
                alpha=0.3,
                label="Train ± SEM",
            )
            ax[1].plot(epochs_axis, va_mcc_mean, label="Val Mean", color="tab:red")
            ax[1].fill_between(
                epochs_axis,
                va_mcc_mean - va_mcc_sem,
                va_mcc_mean + va_mcc_sem,
                color="tab:red",
                alpha=0.3,
                label="Val ± SEM",
            )
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("MCC")
            ax[1].set_title("Aggregated Matthews Correlation Coef")
            ax[1].legend()

            fig.suptitle(
                f"{dset} – {strategy} (n={n_runs} runs)\nLeft: Ground Truth, Right: Generated Samples",
                fontsize=14,
            )
            plt.tight_layout(rect=[0, 0.04, 1, 0.95])

            save_name = f"{dset}_{strategy}_Aggregated_Loss_MCC.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, save_name))
        except Exception as e:
            print(f"Error creating aggregated plot for {strategy}/{dset}: {e}")
        finally:
            plt.close()
