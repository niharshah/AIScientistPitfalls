import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load all experiment_data.npy ----------------
experiment_data_path_list = [
    "None/experiment_data.npy",
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_71d81ca344bf425bb300c42ebc417dc8_proc_1488379/experiment_data.npy",
    "experiments/2025-08-30_20-55-38_gnn_for_spr_attempt_0/logs/0-run/experiment_results/experiment_04829ad17c26424aa90596b063be0cbd_proc_1488378/experiment_data.npy",
]
all_runs = []
for rel_path in experiment_data_path_list:
    try:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), rel_path)
        exp = np.load(abs_path, allow_pickle=True).item()
        all_runs.append(exp)
    except Exception as e:
        print(f"Error loading {rel_path}: {e}")

# ---------------- aggregate SPR_BENCH ----------------
spr_runs = [r["SPR_BENCH"] for r in all_runs if "SPR_BENCH" in r]
if len(spr_runs) == 0:
    print("No SPR_BENCH data found in provided experiment_data.npy files")
else:
    # collect per-epoch curves
    losses_tr_list, losses_val_list = [], []
    acc_tr_list, acc_val_list = [], []
    cowa_tr_list, cowa_val_list = [], []
    final_acc_list, final_cowa_list = [], []

    # helpers (copied from existing plotting code)
    def count_color_variety(sequence: str) -> int:
        return len(
            set(token[1] for token in sequence.strip().split() if len(token) > 1)
        )

    def count_shape_variety(sequence: str) -> int:
        return len(set(token[0] for token in sequence.strip().split() if token))

    def complexity_weight(sequence: str) -> int:
        return count_color_variety(sequence) + count_shape_variety(sequence)

    def complexity_weighted_accuracy(seqs, y_true, y_pred):
        weights = [complexity_weight(s) for s in seqs]
        correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
        return float(sum(correct)) / max(1, sum(weights))

    min_epochs = None
    for data in spr_runs:
        losses_tr_list.append(np.array(data["losses"]["train"]))
        losses_val_list.append(np.array(data["losses"]["val"]))
        acc_tr_list.append(np.array([d["acc"] for d in data["metrics"]["train"]]))
        acc_val_list.append(np.array([d["acc"] for d in data["metrics"]["val"]]))
        cowa_tr_list.append(np.array([d["cowa"] for d in data["metrics"]["train"]]))
        cowa_val_list.append(np.array([d["cowa"] for d in data["metrics"]["val"]]))

        preds = np.array(data["predictions"])
        gts = np.array(data["ground_truth"])
        seqs = data["sequences"]
        final_acc_list.append((preds == gts).mean())
        final_cowa_list.append(complexity_weighted_accuracy(seqs, gts, preds))

        # keep shortest epoch length to align runs
        ep_len = len(data["losses"]["train"])
        min_epochs = ep_len if min_epochs is None else min(min_epochs, ep_len)

    # trim to common length
    losses_tr = np.stack([x[:min_epochs] for x in losses_tr_list])
    losses_val = np.stack([x[:min_epochs] for x in losses_val_list])
    acc_tr = np.stack([x[:min_epochs] for x in acc_tr_list])
    acc_val = np.stack([x[:min_epochs] for x in acc_val_list])
    cowa_tr = np.stack([x[:min_epochs] for x in cowa_tr_list])
    cowa_val = np.stack([x[:min_epochs] for x in cowa_val_list])

    epochs = np.arange(1, min_epochs + 1)

    # --------------- aggregated Loss curve ----------------
    try:
        plt.figure()
        # means
        tr_mean = losses_tr.mean(axis=0)
        val_mean = losses_val.mean(axis=0)
        # stderr
        tr_stderr = losses_tr.std(axis=0) / np.sqrt(losses_tr.shape[0])
        val_stderr = losses_val.std(axis=0) / np.sqrt(losses_val.shape[0])

        plt.plot(epochs, tr_mean, label="Train Loss (mean)", color="tab:blue")
        plt.fill_between(
            epochs,
            tr_mean - tr_stderr,
            tr_mean + tr_stderr,
            color="tab:blue",
            alpha=0.3,
            label="Train Loss ± stderr",
        )
        plt.plot(epochs, val_mean, label="Val Loss (mean)", color="tab:orange")
        plt.fill_between(
            epochs,
            val_mean - val_stderr,
            val_mean + val_stderr,
            color="tab:orange",
            alpha=0.3,
            label="Val Loss ± stderr",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Aggregated Loss Curve\n(mean ± standard error)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve: {e}")
        plt.close()

    # --------------- aggregated Accuracy & CoWA curve ----------------
    try:
        plt.figure()
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        acc_tr_mean = acc_tr.mean(axis=0)
        acc_tr_stderr = acc_tr.std(axis=0) / np.sqrt(acc_tr.shape[0])
        acc_val_mean = acc_val.mean(axis=0)
        acc_val_stderr = acc_val.std(axis=0) / np.sqrt(acc_val.shape[0])

        cowa_tr_mean = cowa_tr.mean(axis=0)
        cowa_tr_stderr = cowa_tr.std(axis=0) / np.sqrt(cowa_tr.shape[0])
        cowa_val_mean = cowa_val.mean(axis=0)
        cowa_val_stderr = cowa_val.std(axis=0) / np.sqrt(cowa_val.shape[0])

        ax1.plot(epochs, acc_tr_mean, "g-", label="Train Acc (mean)")
        ax1.fill_between(
            epochs,
            acc_tr_mean - acc_tr_stderr,
            acc_tr_mean + acc_tr_stderr,
            color="g",
            alpha=0.2,
            label="Train Acc ± stderr",
        )
        ax1.plot(epochs, acc_val_mean, "g--", label="Val Acc (mean)")
        ax1.fill_between(
            epochs,
            acc_val_mean - acc_val_stderr,
            acc_val_mean + acc_val_stderr,
            color="g",
            alpha=0.1,
            label="Val Acc ± stderr",
        )

        ax2.plot(epochs, cowa_tr_mean, "b-", label="Train CoWA (mean)")
        ax2.fill_between(
            epochs,
            cowa_tr_mean - cowa_tr_stderr,
            cowa_tr_mean + cowa_tr_stderr,
            color="b",
            alpha=0.2,
            label="Train CoWA ± stderr",
        )
        ax2.plot(epochs, cowa_val_mean, "b--", label="Val CoWA (mean)")
        ax2.fill_between(
            epochs,
            cowa_val_mean - cowa_val_stderr,
            cowa_val_mean + cowa_val_stderr,
            color="b",
            alpha=0.1,
            label="Val CoWA ± stderr",
        )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy", color="g")
        ax2.set_ylabel("CoWA", color="b")
        plt.title(
            "SPR_BENCH Aggregated Accuracy (green) & CoWA (blue)\n(mean ± standard error)"
        )
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc="center right")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_acc_cowa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated acc/CoWA curve: {e}")
        plt.close()

    # --------------- Final Test Metrics Bar plot ----------------
    try:
        plt.figure()
        metrics = ["Accuracy", "CoWA"]
        means = [np.mean(final_acc_list), np.mean(final_cowa_list)]
        stderrs = [
            np.std(final_acc_list) / np.sqrt(len(final_acc_list)),
            np.std(final_cowa_list) / np.sqrt(len(final_cowa_list)),
        ]
        x = np.arange(len(metrics))
        plt.bar(x, means, yerr=stderrs, capsize=8)
        plt.xticks(x, metrics)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final Test Metrics\n(mean ± standard error)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating final metrics bar plot: {e}")
        plt.close()

    # --------------- print aggregated numbers ----------------
    try:
        print(
            f"Aggregated Final Accuracy: {np.mean(final_acc_list):.3f} ± {np.std(final_acc_list)/np.sqrt(len(final_acc_list)):.3f}"
        )
        print(
            f"Aggregated Final CoWA:     {np.mean(final_cowa_list):.3f} ± {np.std(final_cowa_list)/np.sqrt(len(final_cowa_list)):.3f}"
        )
    except Exception as e:
        print(f"Error printing aggregated metrics: {e}")
