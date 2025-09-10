import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
# basic set-up
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------
# 1) load every run that was passed in "Experiment Data Path"
experiment_data_path_list = [
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_91c7c86a6f894a19896074833877fcd2_proc_2636692/experiment_data.npy",
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_fe2aa7f650fc4637a96918858851b343_proc_2636691/experiment_data.npy",
    "experiments/2025-08-14_12-19-19_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_2fad974d71f94ae29a9452d9bcb644a7_proc_2636690/experiment_data.npy",
]

all_runs = []
for p in experiment_data_path_list:
    try:
        run = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_runs.append(run)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# --------------------------------------------------------------------------
# 2) aggregate per dataset
datasets = set()
for run in all_runs:
    datasets.update(run.keys())

for ds_name in datasets:

    # gather runs that actually contain the dataset
    run_subset = [r[ds_name] for r in all_runs if ds_name in r]
    if len(run_subset) == 0:
        continue

    # helper to stack per-epoch arrays across runs and trim to common length
    def stack_metric(metric_path):
        """metric_path e.g. ('losses','train')"""
        raw = []
        for d in run_subset:
            ptr = d
            ok = True
            for k in metric_path:
                if k not in ptr:
                    ok = False
                    break
                ptr = ptr[k]
            if ok:
                raw.append(np.asarray(ptr))
        if len(raw) == 0:
            return None, None, None  # nothing to plot
        min_len = min(map(len, raw))
        raw = np.stack([r[:min_len] for r in raw], axis=0)  # shape (runs, epochs)
        mean = raw.mean(axis=0)
        se = raw.std(axis=0, ddof=1) / np.sqrt(raw.shape[0])
        epochs = np.arange(1, min_len + 1)
        return epochs, mean, se

    # ----------------------------------------------------------------------
    # 3) LOSS curves (train & dev)
    try:
        ep_tr, m_tr, se_tr = stack_metric(("losses", "train"))
        ep_dev, m_dev, se_dev = stack_metric(("losses", "dev"))
        if m_tr is not None and m_dev is not None:
            plt.figure()
            plt.fill_between(
                ep_tr,
                m_tr - se_tr,
                m_tr + se_tr,
                alpha=0.2,
                color="tab:blue",
                label="Train ±SE",
            )
            plt.plot(ep_tr, m_tr, color="tab:blue", label="Train mean")
            plt.fill_between(
                ep_dev,
                m_dev - se_dev,
                m_dev + se_dev,
                alpha=0.2,
                color="tab:orange",
                label="Dev ±SE",
            )
            plt.plot(ep_dev, m_dev, color="tab:orange", label="Dev mean")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy")
            plt.title(
                f"{ds_name} – Loss Curves (mean ± SE across {len(run_subset)} runs)"
            )
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves_mean_se.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # ----------------------------------------------------------------------
    # 4) ACCURACY curves (train & dev)
    try:
        ep_tr, m_tr, se_tr = stack_metric(("metrics", "train_acc"))
        ep_dev, m_dev, se_dev = stack_metric(("metrics", "dev_acc"))
        if m_tr is not None and m_dev is not None:
            plt.figure()
            plt.fill_between(
                ep_tr,
                m_tr - se_tr,
                m_tr + se_tr,
                alpha=0.2,
                color="tab:blue",
                label="Train ±SE",
            )
            plt.plot(ep_tr, m_tr, color="tab:blue", label="Train mean")
            plt.fill_between(
                ep_dev,
                m_dev - se_dev,
                m_dev + se_dev,
                alpha=0.2,
                color="tab:orange",
                label="Dev ±SE",
            )
            plt.plot(ep_dev, m_dev, color="tab:orange", label="Dev mean")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds_name} – Accuracy Curves (mean ± SE across {len(run_subset)} runs)"
            )
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_accuracy_curves_mean_se.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot for {ds_name}: {e}")
        plt.close()

    # ----------------------------------------------------------------------
    # 5) RGS curve (dev) – if available
    try:
        ep_rgs, m_rgs, se_rgs = stack_metric(("metrics", "dev_rgs"))
        if m_rgs is not None:
            plt.figure()
            plt.fill_between(
                ep_rgs,
                m_rgs - se_rgs,
                m_rgs + se_rgs,
                alpha=0.2,
                color="tab:green",
                label="Dev ±SE",
            )
            plt.plot(ep_rgs, m_rgs, color="tab:green", label="Dev mean")
            plt.ylim(0, 1)
            plt.xlabel("Epoch")
            plt.ylabel("Rule Generalisation Score")
            plt.title(f"{ds_name} – Dev RGS (mean ± SE across {len(run_subset)} runs)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_RGS_curve_mean_se.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated RGS plot for {ds_name}: {e}")
        plt.close()

    # ----------------------------------------------------------------------
    # 6) Confusion matrices (Dev / Test) summed across runs & normalised
    def build_confusion(true, pred):
        n = max(int(np.max(true)), int(np.max(pred))) + 1
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(true, pred):
            cm[t, p] += 1
        return cm

    for split in ["dev", "test"]:
        try:
            sums = None
            for d in run_subset:
                if (
                    "ground_truth" in d
                    and split in d["ground_truth"]
                    and "predictions" in d
                    and split in d["predictions"]
                ):
                    cm = build_confusion(
                        d["ground_truth"][split], d["predictions"][split]
                    )
                    sums = cm if sums is None else sums + cm
            if sums is None:
                continue
            cm_percent = sums / sums.sum() * 100.0
            plt.figure()
            plt.imshow(cm_percent, cmap="viridis")
            plt.title(
                f"{ds_name} – Confusion Matrix ({split.capitalize()}) "
                f"(aggregated over {len(run_subset)} runs)"
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar(label="% of samples")
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_confusion_matrix_{split}_agg.png")
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating aggregated confusion matrix for {ds_name} {split}: {e}"
            )
            plt.close()
