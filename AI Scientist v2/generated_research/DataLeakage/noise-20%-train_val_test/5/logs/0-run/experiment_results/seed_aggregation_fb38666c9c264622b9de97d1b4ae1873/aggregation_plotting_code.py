import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------- #
# basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------- #
# list of experiment-data paths relative to the project root
experiment_data_path_list = [
    "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_d439226fab684c67bdfaa656eca28f4a_proc_3155554/experiment_data.npy",
    "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_9869430d919f429fa00bf8d84c743122_proc_3155553/experiment_data.npy",
    "experiments/2025-08-17_00-45-19_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_69238c2f788443b28891762ef5e7b7e0_proc_3155552/experiment_data.npy",
]

# -------------------------------------------------------------- #
# load all experiments
all_runs = []
for p in experiment_data_path_list:
    try:
        root_path = os.getenv("AI_SCIENTIST_ROOT", "")
        full_path = os.path.join(root_path, p)
        exp = np.load(full_path, allow_pickle=True).item()
        if "SPR_BENCH" in exp:
            all_runs.append(exp["SPR_BENCH"])
    except Exception as e:
        print(f"Error loading {p}: {e}")

n_runs = len(all_runs)
if n_runs == 0:
    print("No valid runs found for SPR_BENCH; aborting plots.")
else:
    # ---------------------------------------------------------- #
    # gather epoch-wise metrics
    train_acc_stack, val_acc_stack = [], []
    train_loss_stack, val_loss_stack = [], []
    preds_all, gts_all, final_test_accs = [], [], []

    # determine common epoch length
    min_epochs = min(len(run["epochs"]) for run in all_runs)

    for run in all_runs:
        # truncate to min_epochs to keep shapes aligned
        train_acc_stack.append(np.asarray(run["metrics"]["train_acc"][:min_epochs]))
        val_acc_stack.append(np.asarray(run["metrics"]["val_acc"][:min_epochs]))
        train_loss_stack.append(np.asarray(run["metrics"]["train_loss"][:min_epochs]))
        val_loss_stack.append(np.asarray(run["metrics"]["val_loss"][:min_epochs]))

        preds_all.extend(run["predictions"])
        gts_all.extend(run["ground_truth"])
        final_test_accs.append(
            float(
                (
                    np.asarray(run["predictions"]) == np.asarray(run["ground_truth"])
                ).mean()
            )
        )

    train_acc_stack = np.vstack(train_acc_stack)
    val_acc_stack = np.vstack(val_acc_stack)
    train_loss_stack = np.vstack(train_loss_stack)
    val_loss_stack = np.vstack(val_loss_stack)

    epochs = np.asarray(all_runs[0]["epochs"][:min_epochs])

    # helper to compute mean & sem
    def mean_sem(arr):
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        return mean, sem

    tr_acc_mean, tr_acc_sem = mean_sem(train_acc_stack)
    val_acc_mean, val_acc_sem = mean_sem(val_acc_stack)
    tr_loss_mean, tr_loss_sem = mean_sem(train_loss_stack)
    val_loss_mean, val_loss_sem = mean_sem(val_loss_stack)

    # ---------------------- Plot 1: accuracy -------------------- #
    try:
        plt.figure()
        plt.plot(epochs, tr_acc_mean, label="Train Acc (mean)")
        plt.fill_between(
            epochs,
            tr_acc_mean - tr_acc_sem,
            tr_acc_mean + tr_acc_sem,
            color="blue",
            alpha=0.2,
            label="Train ± SEM",
        )
        plt.plot(epochs, val_acc_mean, label="Val Acc (mean)")
        plt.fill_between(
            epochs,
            val_acc_mean - val_acc_sem,
            val_acc_mean + val_acc_sem,
            color="orange",
            alpha=0.2,
            label="Val ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR_BENCH Mean Accuracy ± SEM over Epochs\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_mean_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy plot: {e}")
        plt.close()

    # ---------------------- Plot 2: loss ------------------------ #
    try:
        plt.figure()
        plt.plot(epochs, tr_loss_mean, label="Train Loss (mean)")
        plt.fill_between(
            epochs,
            tr_loss_mean - tr_loss_sem,
            tr_loss_mean + tr_loss_sem,
            color="blue",
            alpha=0.2,
            label="Train ± SEM",
        )
        plt.plot(epochs, val_loss_mean, label="Val Loss (mean)")
        plt.fill_between(
            epochs,
            val_loss_mean - val_loss_sem,
            val_loss_mean + val_loss_sem,
            color="orange",
            alpha=0.2,
            label="Val ± SEM",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            "SPR_BENCH Mean Loss ± SEM over Epochs\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_mean_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # ------------------ Plot 3: confusion matrix --------------- #
    try:
        preds_all_np = np.asarray(preds_all)
        gts_all_np = np.asarray(gts_all)
        classes = np.unique(np.concatenate([preds_all_np, gts_all_np]))
        n_cls = len(classes)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds_all_np, gts_all_np):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title(
            "SPR_BENCH Aggregated Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
        )
        fname = os.path.join(working_dir, "SPR_BENCH_aggregated_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated confusion matrix: {e}")
        plt.close()

    # ---------------- Plot 4: final test acc histogram ---------- #
    try:
        plt.figure()
        plt.hist(final_test_accs, bins=min(10, n_runs), alpha=0.7, edgecolor="black")
        plt.xlabel("Final Test Accuracy per Run")
        plt.ylabel("Count")
        plt.title("SPR_BENCH Distribution of Final Test Accuracy Across Runs")
        fname = os.path.join(working_dir, "SPR_BENCH_final_test_accuracy_hist.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final test accuracy histogram: {e}")
        plt.close()

    # ------------------ Print aggregated metric ---------------- #
    final_mean = np.mean(final_test_accs)
    final_sem = np.std(final_test_accs, ddof=1) / np.sqrt(n_runs)
    print(
        f"Final-epoch test accuracy: mean={final_mean:.4f}, SEM={final_sem:.4f} (n={n_runs})"
    )
