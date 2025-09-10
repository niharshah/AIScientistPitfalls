import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- prepare paths & load -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of experiment_data.npy paths extracted from the instruction
experiment_data_path_list = [
    "None/experiment_data.npy",
    "None/experiment_data.npy",
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_3d72018b472146fd9b9ad68c96fdaed5_proc_460563/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        # If AI_SCIENTIST_ROOT env is provided, prepend it; otherwise use path as-is
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        true_path = os.path.join(root, p) if root and not os.path.isabs(p) else p
        exp = np.load(true_path, allow_pickle=True).item()
        all_experiment_data.append(exp)
    except Exception as e:
        print(f"Error loading experiment data from {p}: {e}")

# --------------- aggregation & plotting -----------------
if not all_experiment_data:
    print("No experiment data could be loaded, nothing to plot.")
else:
    # Discover all dataset names that appear in at least one run
    dataset_names = set()
    for exp in all_experiment_data:
        dataset_names.update(exp.keys())

    for dset in dataset_names:
        # ---------- collect per-run arrays ----------
        train_losses_runs, val_losses_runs, val_metric_runs = [], [], []
        for exp in all_experiment_data:
            if dset not in exp:
                continue
            rec = exp[dset]
            if "losses" in rec:
                tr = rec["losses"].get("train")
                vl = rec["losses"].get("val")
                if isinstance(tr, (list, np.ndarray)) and isinstance(
                    vl, (list, np.ndarray)
                ):
                    train_losses_runs.append(np.asarray(tr, dtype=float))
                    val_losses_runs.append(np.asarray(vl, dtype=float))
            if "metrics" in rec:
                vm = rec["metrics"].get("val")
                if isinstance(vm, (list, np.ndarray)):
                    val_metric_runs.append(np.asarray(vm, dtype=float))

        # ---------- plot aggregated loss curves ----------
        try:
            if len(train_losses_runs) >= 2 and len(val_losses_runs) >= 2:
                # Match epoch length across runs
                min_len = min(map(len, train_losses_runs))
                tr_stack = np.stack([x[:min_len] for x in train_losses_runs])
                vl_stack = np.stack([x[:min_len] for x in val_losses_runs])

                tr_mean, tr_se = tr_stack.mean(0), tr_stack.std(0, ddof=1) / np.sqrt(
                    tr_stack.shape[0]
                )
                vl_mean, vl_se = vl_stack.mean(0), vl_stack.std(0, ddof=1) / np.sqrt(
                    vl_stack.shape[0]
                )

                plt.figure()
                epochs = np.arange(min_len)
                plt.plot(epochs, tr_mean, label="Train Mean", color="tab:blue")
                plt.fill_between(
                    epochs,
                    tr_mean - tr_se,
                    tr_mean + tr_se,
                    alpha=0.3,
                    color="tab:blue",
                    label="Train ±SE",
                )
                plt.plot(epochs, vl_mean, label="Val Mean", color="tab:orange")
                plt.fill_between(
                    epochs,
                    vl_mean - vl_se,
                    vl_mean + vl_se,
                    alpha=0.3,
                    color="tab:orange",
                    label="Val ±SE",
                )
                plt.title(
                    f"{dset} Aggregated Loss Curves\n(Mean ± SE across {tr_stack.shape[0]} runs)"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(
                    working_dir, f"{dset}_aggregated_loss_curves.png"
                )
                plt.savefig(save_path)
                plt.close()
            else:
                # not enough runs or data
                pass
        except Exception as e:
            print(f"Error creating aggregated loss plot for {dset}: {e}")
            plt.close()

        # ---------- plot aggregated validation metric ----------
        try:
            if len(val_metric_runs) >= 2:
                min_len_m = min(map(len, val_metric_runs))
                vm_stack = np.stack([x[:min_len_m] for x in val_metric_runs])
                vm_mean = vm_stack.mean(0)
                vm_se = vm_stack.std(0, ddof=1) / np.sqrt(vm_stack.shape[0])

                plt.figure()
                epochs = np.arange(min_len_m)
                plt.plot(epochs, vm_mean, label="Validation Mean", color="tab:green")
                plt.fill_between(
                    epochs,
                    vm_mean - vm_se,
                    vm_mean + vm_se,
                    alpha=0.3,
                    color="tab:green",
                    label="Validation ±SE",
                )
                plt.title(
                    f"{dset} Aggregated Validation Metric\n(Mean ± SE across {vm_stack.shape[0]} runs)"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Metric Value")
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(
                    working_dir, f"{dset}_aggregated_val_metric.png"
                )
                plt.savefig(save_path)
                plt.close()
            else:
                pass
        except Exception as e:
            print(f"Error creating aggregated validation metric plot for {dset}: {e}")
            plt.close()
