import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths & loading -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# List of experiment_data.npy files provided by the user
experiment_data_path_list = [
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_a17205c2281442f3984f3f7888a17e03_proc_458964/experiment_data.npy",
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_1b24efb6a3b04d0085edbd0491e23b20_proc_458965/experiment_data.npy",
    "experiments/2025-07-29_02-18-25_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_de90237d59534abea222b3fc1dcfdab2_proc_458966/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        data = np.load(os.path.join(root, p), allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading experiment data {p}: {e}")

# ------------- aggregate across runs -----------------
# Structure: aggregated[dset]['train_losses'] -> list of np.array, etc.
aggregated = {}
for exp in all_experiment_data:
    for dset, rec in exp.items():
        agg = aggregated.setdefault(
            dset,
            {
                "train_losses": [],
                "val_losses": [],
                "val_metric": [],  # generic holder (e.g. SWA)
                "conf_matrices": [],  # list of confusion matrices
            },
        )
        if "losses" in rec:
            tr = rec["losses"].get("train")
            vl = rec["losses"].get("val")
            if tr is not None and len(tr):
                agg["train_losses"].append(np.asarray(tr))
            if vl is not None and len(vl):
                agg["val_losses"].append(np.asarray(vl))
        if "metrics" in rec:
            vm = rec["metrics"].get("val")
            if vm is not None and len(vm):
                agg["val_metric"].append(np.asarray(vm))
        preds, gts = rec.get("predictions"), rec.get("ground_truth")
        if preds is not None and gts is not None and len(preds):
            labels = sorted(set(gts))
            idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[idx[t], idx[p]] += 1
            agg["conf_matrices"].append(cm)

# ------------- plotting -----------------
for dset, rec in aggregated.items():
    # -------- Figure 1: Aggregated loss curves --------
    try:
        tr_runs, val_runs = rec["train_losses"], rec["val_losses"]
        if tr_runs and val_runs:
            # align lengths
            min_len = min(min(len(r) for r in tr_runs), min(len(r) for r in val_runs))
            tr_stack = np.stack([r[:min_len] for r in tr_runs], axis=0)
            val_stack = np.stack([r[:min_len] for r in val_runs], axis=0)

            tr_mean, tr_se = tr_stack.mean(0), tr_stack.std(0) / np.sqrt(
                tr_stack.shape[0]
            )
            val_mean, val_se = val_stack.mean(0), val_stack.std(0) / np.sqrt(
                val_stack.shape[0]
            )

            epochs = np.arange(min_len)
            plt.figure()
            plt.plot(epochs, tr_mean, "--", label="train mean")
            plt.fill_between(
                epochs, tr_mean - tr_se, tr_mean + tr_se, alpha=0.3, label="train ±SE"
            )
            plt.plot(epochs, val_mean, "-", label="val mean")
            plt.fill_between(
                epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="val ±SE"
            )
            plt.title(
                f"{dset} Aggregated Loss Curves\n(mean ± standard error across runs)"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(working_dir, f"{dset}_aggregated_loss_curves.png")
            plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # -------- Figure 2: Aggregated validation metric --------
    try:
        val_metric_runs = rec["val_metric"]
        if val_metric_runs:
            min_len = min(len(r) for r in val_metric_runs)
            val_stack = np.stack([r[:min_len] for r in val_metric_runs], axis=0)
            val_mean = val_stack.mean(0)
            val_se = val_stack.std(0) / np.sqrt(val_stack.shape[0])
            epochs = np.arange(min_len)
            plt.figure()
            plt.plot(epochs, val_mean, marker="o", label="val mean")
            plt.fill_between(
                epochs, val_mean - val_se, val_mean + val_se, alpha=0.3, label="val ±SE"
            )
            plt.title(f"{dset} Aggregated Validation Metric\n(mean ± standard error)")
            plt.xlabel("Epoch")
            plt.ylabel("Metric value")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(working_dir, f"{dset}_aggregated_val_metric.png")
            plt.savefig(save_path)

            # print final epoch summary
            print(f"{dset}: final val metric = {val_mean[-1]:.4f} ± {val_se[-1]:.4f}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated val metric plot for {dset}: {e}")
        plt.close()

    # -------- Figure 3: Combined confusion matrix --------
    try:
        cms = rec["conf_matrices"]
        if cms:
            combined_cm = sum(cms)
            plt.figure(figsize=(6, 5))
            im = plt.imshow(combined_cm, cmap="Blues")
            plt.colorbar(im)
            plt.title(f"{dset} Combined Confusion Matrix\n(aggregated across runs)")
            num_labels = combined_cm.shape[0]
            labels = np.arange(num_labels)
            plt.xticks(labels, labels, rotation=90, fontsize=6)
            plt.yticks(labels, labels, fontsize=6)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for i in range(num_labels):
                for j in range(num_labels):
                    txt_color = (
                        "white"
                        if combined_cm[i, j] > combined_cm.max() / 2
                        else "black"
                    )
                    plt.text(
                        j,
                        i,
                        combined_cm[i, j],
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontsize=6,
                    )
            plt.tight_layout()
            save_path = os.path.join(
                working_dir, f"{dset}_combined_confusion_matrix.png"
            )
            plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating combined confusion matrix for {dset}: {e}")
        plt.close()
