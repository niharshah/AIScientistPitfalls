import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- specify experiment data paths (update if you have more) ----------
experiment_data_path_list = [
    "experiments/2025-08-31_02-26-55_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_ac40803b3d474d739949b04d022f6e3b_proc_1608774/experiment_data.npy",
    "experiments/2025-08-31_02-26-55_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_06c3f55545a64f08a5b71a98e6d3d00c_proc_1608775/experiment_data.npy",
]  # the "None/experiment_data.npy" entry is skipped because it is invalid

# ---------- load all experiments ----------
all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading {experiment_data_path}: {e}")

if not all_experiment_data:
    print("No experiment data could be loaded – nothing to plot.")
    exit()

# We will only aggregate the SPR_BENCH dataset, but you could extend this
datasets = ["SPR_BENCH"]


def pad_to_max(arrays, pad_value=np.nan):
    """Right-pad 1-D arrays of unequal length with pad_value so they can be stacked."""
    if not arrays:
        return np.array([[]])
    max_len = max(len(a) for a in arrays)
    padded = []
    for a in arrays:
        pad_width = max_len - len(a)
        if pad_width > 0:
            padded.append(np.concatenate([a, np.full(pad_width, pad_value)]))
        else:
            padded.append(np.asarray(a))
    return np.vstack(padded)


for ds in datasets:
    # ----- collect per-run arrays -----
    train_losses, val_losses = [], []
    val_metrics_runs, test_metrics_runs = [], []

    for exp in all_experiment_data:
        exp_ds = exp.get(ds, {})
        if not exp_ds:
            continue
        tl = exp_ds.get("losses", {}).get("train", [])
        vl = exp_ds.get("losses", {}).get("val", [])
        vm = exp_ds.get("metrics", {}).get("val", [])
        tm = exp_ds.get("metrics", {}).get("test", {})
        if tl:
            train_losses.append(np.array(tl, dtype=float))
        if vl:
            val_losses.append(np.array(vl, dtype=float))
        if vm:
            # for easier processing turn list of dicts into dict of lists
            val_metrics_runs.append(vm)
        if tm:
            test_metrics_runs.append(tm)

    # ---------- aggregate losses ----------
    try:
        if train_losses or val_losses:
            epochs = np.arange(1, 1 + max([len(a) for a in train_losses + val_losses]))

            plt.figure()
            if train_losses:
                tl_stack = pad_to_max(train_losses)  # shape: (n_runs, max_len)
                tl_mean = np.nanmean(tl_stack, axis=0)
                tl_se = np.nanstd(tl_stack, axis=0, ddof=1) / np.sqrt(tl_stack.shape[0])
                plt.plot(epochs, tl_mean, label="Train – mean")
                plt.fill_between(
                    epochs,
                    tl_mean - tl_se,
                    tl_mean + tl_se,
                    alpha=0.3,
                    label="Train – SE",
                )

            if val_losses:
                vl_stack = pad_to_max(val_losses)
                vl_mean = np.nanmean(vl_stack, axis=0)
                vl_se = np.nanstd(vl_stack, axis=0, ddof=1) / np.sqrt(vl_stack.shape[0])
                plt.plot(epochs, vl_mean, label="Validation – mean")
                plt.fill_between(
                    epochs,
                    vl_mean - vl_se,
                    vl_mean + vl_se,
                    alpha=0.3,
                    label="Validation – SE",
                )

            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds} Mean ± SE Training/Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_loss_curves_aggregated.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds}: {e}")
        plt.close()

    # ---------- aggregate validation metrics ----------
    try:
        # convert list-of-dict-per-epoch across runs → dict(metric)-> list-of-arrays
        metric_names = ["CWA", "SWA", "GCWA"]
        metric_arrays_per_name = {m: [] for m in metric_names}

        for vm_run in val_metrics_runs:
            # vm_run is list of dicts per epoch
            if not vm_run:
                continue
            for m in metric_names:
                metric_arrays_per_name[m].append(
                    np.array([ep.get(m, np.nan) for ep in vm_run], dtype=float)
                )

        if any(metric_arrays_per_name[m] for m in metric_names):
            max_epochs = max(
                [
                    len(arr)
                    for arrays in metric_arrays_per_name.values()
                    for arr in arrays
                ]
                or [0]
            )
            epochs = np.arange(1, 1 + max_epochs)
            plt.figure()
            colors = dict(CWA="steelblue", SWA="orange", GCWA="green")

            for m in metric_names:
                if metric_arrays_per_name[m]:
                    stack = pad_to_max(metric_arrays_per_name[m])
                    mean = np.nanmean(stack, axis=0)
                    se = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
                    plt.plot(epochs, mean, label=f"{m} – mean", color=colors[m])
                    plt.fill_between(
                        epochs,
                        mean - se,
                        mean + se,
                        alpha=0.3,
                        color=colors[m],
                        label=f"{m} – SE",
                    )

            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.ylim(0, 1)
            plt.title(
                f"{ds} Validation Metrics Mean ± SE\nLeft: CWA, Center: SWA, Right: GCWA"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_validation_metrics_aggregated.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation metric plot for {ds}: {e}")
        plt.close()

    # ---------- aggregate test metrics ----------
    try:
        if test_metrics_runs:
            labels = ["CWA", "SWA", "GCWA"]
            means = []
            ses = []
            for l in labels:
                vals = np.array(
                    [tm.get(l, np.nan) for tm in test_metrics_runs], dtype=float
                )
                means.append(np.nanmean(vals))
                ses.append(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))

            x = np.arange(len(labels))
            plt.figure()
            plt.bar(
                x, means, yerr=ses, capsize=5, color=["steelblue", "orange", "green"]
            )
            plt.xticks(x, labels)
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title(f"{ds} Aggregated Final Test Metrics\nError bars: SE")
            fname = os.path.join(working_dir, f"{ds}_test_metrics_aggregated.png")
            plt.savefig(fname)
            plt.close()

            # print numeric summary
            print(f"{ds} test metrics (mean ± std):")
            for l, m, se in zip(labels, means, ses):
                std = se * np.sqrt(len(test_metrics_runs))
                print(f"  {l}: {m:.4f} ± {std:.4f}")
    except Exception as e:
        print(f"Error creating aggregated test metric bar chart for {ds}: {e}")
        plt.close()
