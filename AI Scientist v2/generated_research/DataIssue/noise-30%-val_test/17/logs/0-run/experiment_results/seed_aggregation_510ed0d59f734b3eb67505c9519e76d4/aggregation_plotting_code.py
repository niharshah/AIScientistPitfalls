import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- locate and load every experiment_data.npy ----------- #
experiment_data_path_list = [
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_35a5b3829b294fd9a13bb73752809195_proc_3335768/experiment_data.npy",
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_83d6fec512a4475cbc3ee316e624f69b_proc_3335769/experiment_data.npy",
    "experiments/2025-08-17_18-47-59_symblic_polyrule_reasoning_attempt_0/logs/0-run/experiment_results/experiment_6a1cfe97917b4d70b089b4428a7ae7a1_proc_3335767/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# -------------- aggregate across runs for each dataset ---------- #
datasets = set()
for ed in all_experiment_data:
    datasets.update(ed.keys())

for dname in datasets:
    # collect per-epoch curves and test metrics
    losses_tr_runs, losses_val_runs = [], []
    mcc_tr_runs, mcc_val_runs = [], []
    test_f1_runs, test_mcc_runs = [], []
    best_val_mcc_runs = []

    for ed in all_experiment_data:
        if dname not in ed:
            continue
        dct = ed[dname]

        # per-epoch curves
        losses_tr_runs.append(np.array(dct["losses"]["train"]))
        losses_val_runs.append(np.array(dct["losses"]["val"]))
        mcc_tr_runs.append(np.array(dct["metrics"]["train"]))
        mcc_val_runs.append(np.array(dct["metrics"]["val"]))

        # test metrics
        preds = np.array(dct["predictions"][0]).flatten()
        gts = np.array(dct["ground_truth"][0]).flatten()
        from sklearn.metrics import f1_score, matthews_corrcoef

        test_f1_runs.append(f1_score(gts, preds, average="macro"))
        test_mcc_runs.append(matthews_corrcoef(gts, preds))

        best_val_mcc_runs.append(float(np.max(dct["metrics"]["val"])))

    # Ensure at least one run exists
    if len(losses_tr_runs) == 0:
        continue

    # truncate to shortest run length so stacking works
    min_len = min(map(len, losses_tr_runs))
    losses_tr = np.stack([x[:min_len] for x in losses_tr_runs])
    losses_val = np.stack([x[:min_len] for x in losses_val_runs])
    mcc_tr = np.stack([x[:min_len] for x in mcc_tr_runs])
    mcc_val = np.stack([x[:min_len] for x in mcc_val_runs])

    epochs = np.arange(min_len)

    # ---------------- aggregated LOSS plot ----------------------- #
    try:
        plt.figure()
        mean_tr, sem_tr = losses_tr.mean(0), losses_tr.std(0) / np.sqrt(
            losses_tr.shape[0]
        )
        mean_val, sem_val = losses_val.mean(0), losses_val.std(0) / np.sqrt(
            losses_val.shape[0]
        )

        plt.plot(epochs, mean_tr, label="Train Mean", color="blue")
        plt.fill_between(
            epochs,
            mean_tr - sem_tr,
            mean_tr + sem_tr,
            color="blue",
            alpha=0.2,
            label="Train ± SEM",
        )

        plt.plot(epochs, mean_val, label="Val Mean", color="red")
        plt.fill_between(
            epochs,
            mean_val - sem_val,
            mean_val + sem_val,
            color="red",
            alpha=0.2,
            label="Val ± SEM",
        )

        plt.title(
            f"{dname} Aggregated Loss Curves\nMean ± SEM over {losses_tr.shape[0]} runs"
        )
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_agg_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot ({dname}): {e}")
        plt.close()

    # ---------------- aggregated MCC plot ------------------------ #
    try:
        plt.figure()
        mean_tr, sem_tr = mcc_tr.mean(0), mcc_tr.std(0) / np.sqrt(mcc_tr.shape[0])
        mean_val, sem_val = mcc_val.mean(0), mcc_val.std(0) / np.sqrt(mcc_val.shape[0])

        plt.plot(epochs, mean_tr, label="Train Mean", color="green")
        plt.fill_between(
            epochs,
            mean_tr - sem_tr,
            mean_tr + sem_tr,
            color="green",
            alpha=0.2,
            label="Train ± SEM",
        )

        plt.plot(epochs, mean_val, label="Val Mean", color="orange")
        plt.fill_between(
            epochs,
            mean_val - sem_val,
            mean_val + sem_val,
            color="orange",
            alpha=0.2,
            label="Val ± SEM",
        )

        plt.title(
            f"{dname} Aggregated MCC Curves\nMean ± SEM over {mcc_tr.shape[0]} runs"
        )
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_agg_mcc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated MCC plot ({dname}): {e}")
        plt.close()

    # ---------------- aggregated Test Metrics -------------------- #
    try:
        mean_f1 = np.mean(test_f1_runs)
        sem_f1 = np.std(test_f1_runs) / np.sqrt(len(test_f1_runs))
        mean_mcc = np.mean(test_mcc_runs)
        sem_mcc = np.std(test_mcc_runs) / np.sqrt(len(test_mcc_runs))

        plt.figure()
        bars = plt.bar(
            ["Macro-F1", "MCC"],
            [mean_f1, mean_mcc],
            yerr=[sem_f1, sem_mcc],
            color=["steelblue", "orange"],
            capsize=5,
            label="Mean ± SEM",
        )
        plt.ylim(0, 1)
        plt.title(
            f"{dname} Aggregated Test Metrics\nMean ± SEM over {len(test_f1_runs)} runs"
        )
        for bar, v in zip(bars, [mean_f1, mean_mcc]):
            plt.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center"
            )
        plt.legend()
        fname = os.path.join(working_dir, f"{dname.lower()}_agg_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated test metrics ({dname}): {e}")
        plt.close()

    # save mean best val mcc for cross-dataset comparison
    globals().setdefault("best_val_mcc_all_mean", {})[dname] = (
        np.mean(best_val_mcc_runs),
        np.std(best_val_mcc_runs) / np.sqrt(len(best_val_mcc_runs)),
    )

# ------------ Cross-dataset comparison (best val MCC) ----------- #
try:
    if "best_val_mcc_all_mean" in globals() and best_val_mcc_all_mean:
        names = list(best_val_mcc_all_mean.keys())
        means = [best_val_mcc_all_mean[n][0] for n in names]
        sems = [best_val_mcc_all_mean[n][1] for n in names]

        plt.figure()
        bars = plt.bar(
            names, means, yerr=sems, color="purple", capsize=5, label="Mean ± SEM"
        )
        plt.ylim(0, 1)
        plt.title("Best Validation MCC Across Datasets\nMean ± SEM over runs")
        for bar, v in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center"
            )
        plt.legend()
        fname = os.path.join(working_dir, "comparison_best_val_mcc_agg.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating aggregated cross-dataset comparison: {e}")
    plt.close()
