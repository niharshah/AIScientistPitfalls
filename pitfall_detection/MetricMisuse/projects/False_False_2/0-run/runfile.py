import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------------------
# 1. LOAD ALL RUNS ------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_fdf743c8f8d64251bec6fe4e9099d3a7_proc_335107/experiment_data.npy",
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_d7cd79645a7c41848f0e8bd15748253f_proc_335108/experiment_data.npy",
    "experiments/2025-07-28_01-00-31_neural_symbolic_zero_shot_spr_attempt_0/logs/0-run/experiment_results/experiment_39d0aad81bed43ebb2cc9ddffa837aa6_proc_335109/experiment_data.npy",
]

all_exp_data = []
for p in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        data = np.load(full_path, allow_pickle=True).item()
        all_exp_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ----------------------------------------------------------------------
# 2. COLLECT CURVES -----------------------------------------------------
curves = {
    "epochs": [],
    "train_loss": [],
    "dev_loss": [],
    "train_pha": [],
    "dev_pha": [],
}
test_metrics_runs = []

for d in all_exp_data:
    try:
        exp = d["remove_shape_features"]["spr_bench"]
    except Exception:
        continue

    # epochs (assume identical length; if not, truncate to the min later)
    curves["epochs"].append(np.asarray(exp.get("epochs", [])))
    curves["train_loss"].append(
        np.asarray(exp.get("losses", {}).get("train", []), dtype=float)
    )
    curves["dev_loss"].append(
        np.asarray(exp.get("losses", {}).get("dev", []), dtype=float)
    )
    curves["train_pha"].append(
        np.asarray(exp.get("metrics", {}).get("train_PHA", []), dtype=float)
    )
    curves["dev_pha"].append(
        np.asarray(exp.get("metrics", {}).get("dev_PHA", []), dtype=float)
    )
    test_metrics_runs.append(exp.get("test_metrics", {}))

# Make sure we have data
if len(curves["epochs"]) == 0:
    print("No valid runs found to aggregate.")
else:
    # Determine minimal length for epoch-wise stacking
    min_len = min([len(e) for e in curves["epochs"]])
    epochs_ref = curves["epochs"][0][:min_len]

    def stack_and_crop(key):
        arr = [x[:min_len] for x in curves[key] if len(x) >= min_len]
        return np.vstack(arr) if len(arr) > 0 else None

    train_loss_stack = stack_and_crop("train_loss")
    dev_loss_stack = stack_and_crop("dev_loss")
    train_pha_stack = stack_and_crop("train_pha")
    dev_pha_stack = stack_and_crop("dev_pha")

    # ------------------------------------------------------------------
    # 3. PLOT LOSS ------------------------------------------------------
    try:
        plt.figure()
        if train_loss_stack is not None:
            mean = train_loss_stack.mean(axis=0)
            stderr = train_loss_stack.std(axis=0) / np.sqrt(train_loss_stack.shape[0])
            plt.plot(epochs_ref, mean, label="Train (mean)")
            plt.fill_between(epochs_ref, mean - stderr, mean + stderr, alpha=0.3)
        if dev_loss_stack is not None:
            mean = dev_loss_stack.mean(axis=0)
            stderr = dev_loss_stack.std(axis=0) / np.sqrt(dev_loss_stack.shape[0])
            plt.plot(epochs_ref, mean, label="Validation (mean)")
            plt.fill_between(epochs_ref, mean - stderr, mean + stderr, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Color-Only Loss Curves (Aggregated)")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "spr_bench_color_only_loss_curves_aggregated.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 4. PLOT PHA -------------------------------------------------------
    try:
        plt.figure()
        if train_pha_stack is not None:
            mean = train_pha_stack.mean(axis=0)
            stderr = train_pha_stack.std(axis=0) / np.sqrt(train_pha_stack.shape[0])
            plt.plot(epochs_ref, mean, label="Train PHA (mean)")
            plt.fill_between(epochs_ref, mean - stderr, mean + stderr, alpha=0.3)
        if dev_pha_stack is not None:
            mean = dev_pha_stack.mean(axis=0)
            stderr = dev_pha_stack.std(axis=0) / np.sqrt(dev_pha_stack.shape[0])
            plt.plot(epochs_ref, mean, label="Validation PHA (mean)")
            plt.fill_between(epochs_ref, mean - stderr, mean + stderr, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.title("SPR_BENCH Color-Only PHA Curves (Aggregated)")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "spr_bench_color_only_pha_curves_aggregated.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated PHA plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # 5. PLOT TEST METRICS ---------------------------------------------
    try:
        metric_names = ["SWA", "CWA", "PHA"]
        values = {k: [] for k in metric_names}
        for tm in test_metrics_runs:
            for k in metric_names:
                if k in tm:
                    values[k].append(tm[k])

        means = np.array(
            [np.mean(values[k]) if len(values[k]) else np.nan for k in metric_names]
        )
        stderrs = np.array(
            [
                (
                    np.std(values[k]) / np.sqrt(len(values[k]))
                    if len(values[k])
                    else np.nan
                )
                for k in metric_names
            ]
        )

        plt.figure()
        x_pos = np.arange(len(metric_names))
        plt.bar(
            x_pos,
            means,
            yerr=stderrs,
            capsize=5,
            color=["tab:blue", "tab:orange", "tab:green"],
        )
        plt.xticks(x_pos, metric_names)
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Color-Only Test Metrics (Aggregated)")
        for i, v in enumerate(means):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.legend(["± StdErr"])
        plt.savefig(
            os.path.join(
                working_dir, "spr_bench_color_only_test_metrics_aggregated.png"
            )
        )
        plt.close()

        # print aggregated numbers
        print("Aggregated test metrics (mean ± stderr):")
        for k, m, s in zip(metric_names, means, stderrs):
            print(f"{k}: {m:.3f} ± {s:.3f}")
    except Exception as e:
        print(f"Error creating aggregated test metrics plot: {e}")
        plt.close()
