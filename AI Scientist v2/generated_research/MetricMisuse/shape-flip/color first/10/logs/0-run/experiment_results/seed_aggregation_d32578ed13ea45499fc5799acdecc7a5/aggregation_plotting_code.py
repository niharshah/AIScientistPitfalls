import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# Set up working directory
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# Collect all experiment_data dictionaries
# ------------------------------------------------------------
try:
    # Paths provided in the "Experiment Data Path" section
    experiment_data_path_list = [
        "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_af25fa889c504cd7895ce35ef07dc8cc_proc_1544298/experiment_data.npy",
        "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_b84393766c71426599835acf6e80ada8_proc_1544300/experiment_data.npy",
        "experiments/2025-08-30_23-24-25_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_00c6e23a0e4348779bdfe1cc20c26c85_proc_1544299/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if os.path.isfile(full_path):
            all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
        else:
            print(f"File not found: {full_path}")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ------------------------------------------------------------
# Helper for mean and standard error
# ------------------------------------------------------------
def mean_sem(arrays):
    """Stack 1-D arrays to compute mean and standard error along axis 0."""
    stack = np.stack(arrays, axis=0)
    mean = stack.mean(axis=0)
    sem = (
        stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        if stack.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# ------------------------------------------------------------
# Aggregate and plot for every dataset present
# ------------------------------------------------------------
datasets = set()
for ed in all_experiment_data:
    datasets.update(ed.get("EPOCHS", {}).keys())

for ds in sorted(datasets):
    # Collect per-run time-series across all files
    train_losses, val_losses, val_hwas = [], [], []
    run_final_hwa = {}  # run_key -> list of values across files
    test_hwas = []  # aggregated test HWA over files

    for ed in all_experiment_data:
        runs_dict = ed.get("EPOCHS", {}).get(ds, {})
        # Store overall test metric if present (once per file)
        if "metrics_test" in runs_dict and "hwa" in runs_dict["metrics_test"]:
            test_hwas.append(runs_dict["metrics_test"]["hwa"])
        # Iterate runs
        for rk, rv in runs_dict.items():
            if not rk.startswith("run_"):
                continue
            # losses
            if "losses" in rv:
                tl = np.asarray(rv["losses"]["train"])
                vl = np.asarray(rv["losses"]["val"])
                train_losses.append(tl)
                val_losses.append(vl)
            # val metrics
            if "metrics" in rv and "val" in rv["metrics"]:
                hwa_series = [m["hwa"] for m in rv["metrics"]["val"]]
                val_hwas.append(np.asarray(hwa_series))
            # final HWA
            if "final_val_hwa" in rv:
                run_final_hwa.setdefault(rk, []).append(rv["final_val_hwa"])

    # --------------------------------------------------------
    # Align lengths (truncate to minimum) for loss & HWA curves
    # --------------------------------------------------------
    if train_losses:
        min_len_loss = min(map(len, train_losses + val_losses))
        train_losses = [tl[:min_len_loss] for tl in train_losses]
        val_losses = [vl[:min_len_loss] for vl in val_losses]

    if val_hwas:
        min_len_hwa = min(map(len, val_hwas))
        val_hwas = [vh[:min_len_hwa] for vh in val_hwas]

    # --------------------------------------------------------
    # 1. Aggregated loss curves
    # --------------------------------------------------------
    try:
        if train_losses:
            epochs = np.arange(1, min_len_loss + 1)
            m_tr, se_tr = mean_sem(train_losses)
            m_val, se_val = mean_sem(val_losses)

            plt.figure()
            plt.plot(epochs, m_tr, label="Train Loss (mean)")
            plt.fill_between(
                epochs, m_tr - se_tr, m_tr + se_tr, alpha=0.3, label="Train SEM"
            )
            plt.plot(epochs, m_val, label="Val Loss (mean)")
            plt.fill_between(
                epochs, m_val - se_val, m_val + se_val, alpha=0.3, label="Val SEM"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{ds}: Aggregated Loss Curves\nMean ± SEM over {len(train_losses)} runs"
            )
            plt.legend()
            fname = f"{ds.lower()}_aggregated_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds}: {e}")
        plt.close()

    # --------------------------------------------------------
    # 2. Aggregated validation HWA curves
    # --------------------------------------------------------
    try:
        if val_hwas:
            epochs = np.arange(1, min_len_hwa + 1)
            m_hwa, se_hwa = mean_sem(val_hwas)
            plt.figure()
            plt.plot(epochs, m_hwa, marker="o", label="Val HWA (mean)")
            plt.fill_between(
                epochs, m_hwa - se_hwa, m_hwa + se_hwa, alpha=0.3, label="SEM"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Harmonic Weighted Acc")
            plt.title(
                f"{ds}: Aggregated Validation HWA\nMean ± SEM over {len(val_hwas)} runs"
            )
            plt.legend()
            fname = f"{ds.lower()}_aggregated_val_hwa.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated HWA plot for {ds}: {e}")
        plt.close()

    # --------------------------------------------------------
    # 3. Final validation HWA per run (bar with error bars)
    # --------------------------------------------------------
    try:
        if run_final_hwa:
            run_names = sorted(
                run_final_hwa.keys(), key=lambda s: int(s.split("_")[-1])
            )
            means = [np.mean(run_final_hwa[r]) for r in run_names]
            sems = [
                (
                    np.std(run_final_hwa[r], ddof=1) / np.sqrt(len(run_final_hwa[r]))
                    if len(run_final_hwa[r]) > 1
                    else 0.0
                )
                for r in run_names
            ]

            plt.figure()
            plt.bar(run_names, means, yerr=sems, capsize=5)
            plt.ylabel("Final Val HWA")
            plt.title(
                f"{ds}: Final Validation HWA by Run\nMean ± SEM over {len(all_experiment_data)} files"
            )
            plt.xticks(rotation=45)
            plt.tight_layout()
            fname = f"{ds.lower()}_final_val_hwa_bar.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating final HWA bar chart for {ds}: {e}")
        plt.close()

    # --------------------------------------------------------
    # 4. Print aggregated test metrics
    # --------------------------------------------------------
    try:
        if test_hwas:
            mean_test_hwa = np.mean(test_hwas)
            sem_test_hwa = (
                (np.std(test_hwas, ddof=1) / np.sqrt(len(test_hwas)))
                if len(test_hwas) > 1
                else 0.0
            )
            print(
                f"{ds} TEST HWA -> mean: {mean_test_hwa:.3f}, SEM: {sem_test_hwa:.3f}  (n={len(test_hwas)})"
            )
    except Exception as e:
        print(f"Error printing test metrics for {ds}: {e}")
