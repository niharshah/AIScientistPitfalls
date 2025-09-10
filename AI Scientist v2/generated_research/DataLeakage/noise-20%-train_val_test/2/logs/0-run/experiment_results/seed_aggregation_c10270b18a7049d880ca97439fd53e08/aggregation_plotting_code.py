import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------- paths -------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- list of experiment_data.npy to aggregate -------
experiment_data_path_list = [
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_5c8ffee349004c9fa084a6bbbf28a902_proc_3164417/experiment_data.npy",
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_6c09d20f8857496686af4a00a186a115_proc_3164418/experiment_data.npy",
    "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_af340dac1c35422682b75bdc8305c60d_proc_3164419/experiment_data.npy",
]

# ------------------ load all experiment data -------------
all_runs = []  # list of dicts, each dict is one "run" (a key inside a file)
epochs_list = []  # to record epoch arrays for alignment

try:
    for exp_path in experiment_data_path_list:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        full_path = os.path.join(root, exp_path) if root else exp_path
        data = np.load(full_path, allow_pickle=True).item()
        for k in data.keys():  # flatten any inner keys
            run = data[k]
            run["name"] = k  # keep original name
            all_runs.append(run)
            epochs_list.append(np.asarray(run.get("epochs", [])))
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_runs, epochs_list = [], []

if len(all_runs) == 0:
    print("No runs found – nothing to plot.")
else:
    # Align epochs by taking the minimum common length
    min_len = min([len(ep) for ep in epochs_list])
    epochs = epochs_list[0][:min_len]

    # --------------- helper to stack and aggregate ----------------
    def stack_metric(metric_path):
        """metric_path example: ('metrics','train_macro_f1')"""
        collected = []
        for run in all_runs:
            entry = run
            try:
                for key in metric_path:
                    entry = entry[key]
                collected.append(np.asarray(entry)[:min_len])
            except Exception:
                pass
        return np.asarray(collected)  # shape (n_runs, min_len)

    # ------------------- 1) Macro-F1 curves ------------------------
    try:
        tr_f1_arr = stack_metric(("metrics", "train_macro_f1"))
        val_f1_arr = stack_metric(("metrics", "val_macro_f1"))

        plt.figure()
        # plot individual runs in faint colors
        for i, arr in enumerate(tr_f1_arr):
            plt.plot(epochs, arr, "--", alpha=0.3, label=f"run{i}-train")
        for i, arr in enumerate(val_f1_arr):
            plt.plot(epochs, arr, "-", alpha=0.3, label=f"run{i}-val")

        # aggregated mean & stderr
        if tr_f1_arr.size and val_f1_arr.size:
            for label, arr, style in [
                ("Train", tr_f1_arr, "--"),
                ("Val", val_f1_arr, "-"),
            ]:
                mean = arr.mean(axis=0)
                stderr = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                plt.plot(
                    epochs,
                    mean,
                    style,
                    color="black",
                    linewidth=2,
                    label=f"Mean-{label}",
                )
                plt.fill_between(
                    epochs,
                    mean - stderr,
                    mean + stderr,
                    alpha=0.2,
                    color="black",
                    label=f"SEM-{label}",
                )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 (Mean ± SEM, individual runs faint)")
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_macro_f1_mean_sem.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated Macro-F1 plot: {e}")
        plt.close()

    # --------------------- 2) Loss curves -------------------------
    try:
        tr_loss_arr = stack_metric(("losses", "train"))
        val_loss_arr = stack_metric(("losses", "val"))

        plt.figure()
        for i, arr in enumerate(tr_loss_arr):
            plt.plot(epochs, arr, "--", alpha=0.3)
        for i, arr in enumerate(val_loss_arr):
            plt.plot(epochs, arr, "-", alpha=0.3)

        if tr_loss_arr.size and val_loss_arr.size:
            for label, arr, style in [
                ("Train", tr_loss_arr, "--"),
                ("Val", val_loss_arr, "-"),
            ]:
                mean = arr.mean(axis=0)
                stderr = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                plt.plot(
                    epochs,
                    mean,
                    style,
                    color="black",
                    linewidth=2,
                    label=f"Mean-{label}",
                )
                plt.fill_between(
                    epochs,
                    mean - stderr,
                    mean + stderr,
                    alpha=0.2,
                    color="black",
                    label=f"SEM-{label}",
                )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss (Mean ± SEM)")
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_mean_sem.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated Loss plot: {e}")
        plt.close()

    # --------------- 3) Test Macro-F1 bar chart ------------------
    try:
        test_scores = []
        run_labels = []
        for run in all_runs:
            if "test_macro_f1" in run:
                test_scores.append(run["test_macro_f1"])
                run_labels.append(run["name"])
        test_scores = np.asarray(test_scores)
        plt.figure()
        x = np.arange(len(test_scores))
        plt.bar(x, test_scores, tick_label=run_labels)
        mean = test_scores.mean()
        stderr = (
            test_scores.std(ddof=1) / np.sqrt(len(test_scores))
            if len(test_scores) > 1
            else 0
        )
        plt.axhline(mean, color="red", linewidth=2, label="Mean")
        plt.fill_between(
            [-0.5, len(test_scores) - 0.5],
            mean - stderr,
            mean + stderr,
            color="red",
            alpha=0.2,
            label="SEM",
        )
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Test Macro-F1 (runs + mean ± SEM)")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "spr_bench_test_macro_f1_runs_and_mean.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating Test Macro-F1 summary bar: {e}")
        plt.close()

    # -------- 4) Confusion matrices for first ≤5 runs -------------
    try:
        from sklearn.metrics import confusion_matrix

        plotted = 0
        for run in all_runs:
            if plotted >= 5:
                break
            preds = run.get("predictions")
            gts = run.get("ground_truth")
            if preds is None or gts is None or len(preds) == 0:
                continue
            cm = confusion_matrix(gts, preds)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{run['name']} Confusion Matrix (Test Set)")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"spr_bench_{run['name']}_confusion_matrix.png"
                )
            )
            plt.close()
            plotted += 1
    except Exception as e:
        print(f"Error creating Confusion Matrices: {e}")
        plt.close()

    # ---------------- print numeric summary ----------------------
    print("Individual Test Macro-F1:", dict(zip(run_labels, test_scores)))
    if len(test_scores):
        print(f"Mean Test Macro-F1: {mean:.4f} ± {stderr:.4f} (SEM)")
