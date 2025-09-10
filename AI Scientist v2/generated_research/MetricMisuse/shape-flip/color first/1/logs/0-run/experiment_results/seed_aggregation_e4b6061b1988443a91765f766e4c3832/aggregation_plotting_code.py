import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load every run that is available
experiment_data_path_list = [
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_e957cdc00a39422380321c45cb13e2ee_proc_1449034/experiment_data.npy",
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_af637211023c43aca09e70d9a79c73f8_proc_1449036/experiment_data.npy",
    "experiments/2025-08-30_17-49-30_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_7341cdd69955463fa3e561a90fb83797_proc_1449035/experiment_data.npy",
]

all_experiments = []
for path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), path)
        exp = np.load(full_path, allow_pickle=True).item()
        all_experiments.append(exp)
        print(f"Loaded {full_path}")
    except Exception as e:
        print(f"Error loading {path}: {e}")


# ------------------------------------------------------------------
def collect_field(exp_list, outer_key, *field_chain):
    """
    Grab a nested list/array from every experiment and return as list.
    Example: collect_field(all_experiments, 'SPR', 'losses', 'train')
    """
    out = []
    for e in exp_list:
        try:
            d = e[outer_key]
            for f in field_chain:
                d = d[f]
            out.append(np.asarray(d))
        except Exception:
            pass
    return out


if not all_experiments:
    print("No experiment data could be loaded – aborting plots.")
else:
    # we only aggregate the dataset key that exists in the first run
    dataset_key = list(all_experiments[0].keys())[0]  # 'SPR'
    runs_train_loss = collect_field(all_experiments, dataset_key, "losses", "train")
    runs_val_loss = collect_field(all_experiments, dataset_key, "losses", "val")
    runs_metrics_val = collect_field(all_experiments, dataset_key, "metrics", "val")

    # to stack arrays they must have same length – truncate to shortest run
    min_len = min(map(len, runs_train_loss + runs_val_loss))
    runs_train_loss = [x[:min_len] for x in runs_train_loss]
    runs_val_loss = [x[:min_len] for x in runs_val_loss]

    # build numpy arrays of shape [n_runs, min_len]
    train_loss_np = np.vstack(runs_train_loss)
    val_loss_np = np.vstack(runs_val_loss)
    epochs = np.arange(1, min_len + 1)

    # helper to extract metric curves
    def metric_matrix(metric_name):
        curves = []
        for run in runs_metrics_val:
            # each run is list of dicts per epoch
            vals = [ep.get(metric_name, np.nan) for ep in run[:min_len]]
            curves.append(vals)
        return np.vstack(curves)

    # ------------------------------------------------------------------
    # Aggregated Train vs Validation loss
    try:
        plt.figure()
        mu_train = train_loss_np.mean(axis=0)
        sem_train = train_loss_np.std(axis=0, ddof=1) / np.sqrt(train_loss_np.shape[0])
        mu_val = val_loss_np.mean(axis=0)
        sem_val = val_loss_np.std(axis=0, ddof=1) / np.sqrt(val_loss_np.shape[0])

        plt.plot(epochs, mu_train, label="Train mean", color="tab:blue")
        plt.fill_between(
            epochs,
            mu_train - sem_train,
            mu_train + sem_train,
            color="tab:blue",
            alpha=0.3,
            label="Train ± SEM",
        )
        plt.plot(epochs, mu_val, label="Validation mean", color="tab:orange")
        plt.fill_between(
            epochs,
            mu_val - sem_val,
            mu_val + sem_val,
            color="tab:orange",
            alpha=0.3,
            label="Val ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_key} – Aggregated Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset_key}_aggregated_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Aggregated Accuracy & HPA
    try:
        acc_mat = metric_matrix("acc")
        hpa_mat = metric_matrix("HPA")
        if acc_mat.size and hpa_mat.size:
            mu_acc = np.nanmean(acc_mat, axis=0)
            sem_acc = np.nanstd(acc_mat, axis=0, ddof=1) / np.sqrt(acc_mat.shape[0])
            mu_hpa = np.nanmean(hpa_mat, axis=0)
            sem_hpa = np.nanstd(hpa_mat, axis=0, ddof=1) / np.sqrt(hpa_mat.shape[0])

            plt.figure()
            plt.plot(epochs, mu_acc, label="Accuracy mean", color="tab:green")
            plt.fill_between(
                epochs,
                mu_acc - sem_acc,
                mu_acc + sem_acc,
                color="tab:green",
                alpha=0.3,
                label="Acc ± SEM",
            )
            plt.plot(epochs, mu_hpa, label="HPA mean", color="tab:red")
            plt.fill_between(
                epochs,
                mu_hpa - sem_hpa,
                mu_hpa + sem_hpa,
                color="tab:red",
                alpha=0.3,
                label="HPA ± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dataset_key} – Aggregated Validation Accuracy & HPA")
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset_key}_aggregated_acc_hpa.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        else:
            print("Accuracy or HPA not found in metrics – skipping acc/HPA plot.")
    except Exception as e:
        print(f"Error creating aggregated acc/HPA plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Aggregated CWA & SWA
    try:
        cwa_mat = metric_matrix("CWA")
        swa_mat = metric_matrix("SWA")
        if cwa_mat.size and swa_mat.size:
            mu_cwa = np.nanmean(cwa_mat, axis=0)
            sem_cwa = np.nanstd(cwa_mat, axis=0, ddof=1) / np.sqrt(cwa_mat.shape[0])
            mu_swa = np.nanmean(swa_mat, axis=0)
            sem_swa = np.nanstd(swa_mat, axis=0, ddof=1) / np.sqrt(swa_mat.shape[0])

            plt.figure()
            plt.plot(epochs, mu_cwa, label="CWA mean", color="tab:purple")
            plt.fill_between(
                epochs,
                mu_cwa - sem_cwa,
                mu_cwa + sem_cwa,
                color="tab:purple",
                alpha=0.3,
                label="CWA ± SEM",
            )
            plt.plot(epochs, mu_swa, label="SWA mean", color="tab:brown")
            plt.fill_between(
                epochs,
                mu_swa - sem_swa,
                mu_swa + sem_swa,
                color="tab:brown",
                alpha=0.3,
                label="SWA ± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dataset_key} – Aggregated CWA & SWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset_key}_aggregated_cwa_swa.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        else:
            print("CWA or SWA not found in metrics – skipping CWA/SWA plot.")
    except Exception as e:
        print(f"Error creating aggregated CWA/SWA plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Print summary metric
    try:
        if acc_mat.size:
            final_accs = acc_mat[:, -1]
            print(
                f"Final-epoch Validation Accuracy: mean={final_accs.mean():.4f}, "
                f"SEM={final_accs.std(ddof=1)/np.sqrt(len(final_accs)):.4f}, "
                f"N={len(final_accs)}"
            )
    except Exception as e:
        print(f"Error printing summary metric: {e}")
