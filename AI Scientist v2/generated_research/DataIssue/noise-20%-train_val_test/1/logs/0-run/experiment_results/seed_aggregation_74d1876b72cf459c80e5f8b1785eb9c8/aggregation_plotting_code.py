import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- set-up and load multiple runs ----------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Paths supplied in the instructions
experiment_data_path_list = [
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_6aa66816593f4e348669a73aefe3accf_proc_3165650/experiment_data.npy",
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_f98776920b86441fbe3539396a25ac23_proc_3165651/experiment_data.npy",
    "experiments/2025-08-17_00-43-58_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_50e9a1032a074217aeb86db5fd47fd2e_proc_3165649/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp = np.load(full_p, allow_pickle=True).item()
        if "SPR_BENCH" in exp:
            all_experiment_data.append(exp["SPR_BENCH"])
        else:
            print(f"Warning: 'SPR_BENCH' not found in {p}, skipping this run.")
    except Exception as e:
        print(f"Error loading {p}: {e}")

if not all_experiment_data:
    print("No valid SPR_BENCH runs found; nothing to aggregate.")
    exit()


# --------------------- helper to extract arrays safely -------------------- #
def _np(arr_name, d, parent_key, fallback=[]):
    return np.asarray(d.get(parent_key, {}).get(arr_name, fallback), dtype=float)


# Gather per-run arrays
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_f1s, val_f1s = [], []
test_preds, test_gts = [], []
for run in all_experiment_data:
    train_losses.append(_np("train", run, "losses"))
    val_losses.append(_np("val", run, "losses"))
    train_accs.append(_np("train_acc", run, "metrics"))
    val_accs.append(_np("val_acc", run, "metrics"))
    train_f1s.append(_np("train_f1", run, "metrics"))
    val_f1s.append(_np("val_f1", run, "metrics"))
    # predictions/gt could be absent
    test_preds.append(np.asarray(run.get("predictions", []), dtype=int))
    test_gts.append(np.asarray(run.get("ground_truth", []), dtype=int))

# Trim to common epoch length
min_epochs = min(map(len, train_losses))


def _stack_and_trim(lst):
    return np.stack([x[:min_epochs] for x in lst], axis=0)  # shape (runs, epochs)


train_losses = _stack_and_trim(train_losses)
val_losses = _stack_and_trim(val_losses)
train_accs = _stack_and_trim(train_accs)
val_accs = _stack_and_trim(val_accs)
train_f1s = _stack_and_trim(train_f1s)
val_f1s = _stack_and_trim(val_f1s)
epochs = np.arange(1, min_epochs + 1)


# Mean and SE
def _mean_se(arr):
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])  # standard error
    return mean, se


mean_train_loss, se_train_loss = _mean_se(train_losses)
mean_val_loss, se_val_loss = _mean_se(val_losses)
mean_train_acc, se_train_acc = _mean_se(train_accs)
mean_val_acc, se_val_acc = _mean_se(val_accs)
mean_train_f1, se_train_f1 = _mean_se(train_f1s)
mean_val_f1, se_val_f1 = _mean_se(val_f1s)


def _safe_save(fig_name):
    return os.path.join(working_dir, f"spr_bench_{fig_name}.png")


# ----------------------------- 1. Loss Curves ------------------------------ #
try:
    plt.figure()
    plt.plot(epochs, mean_train_loss, label="Train (mean)")
    plt.fill_between(
        epochs,
        mean_train_loss - se_train_loss,
        mean_train_loss + se_train_loss,
        alpha=0.3,
        label="Train (±SE)",
    )
    plt.plot(epochs, mean_val_loss, label="Validation (mean)")
    plt.fill_between(
        epochs,
        mean_val_loss - se_val_loss,
        mean_val_loss + se_val_loss,
        alpha=0.3,
        label="Validation (±SE)",
    )
    plt.title("SPR_BENCH Loss Curves with Mean ± SE (Sequence Classification)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig(_safe_save("mean_loss_curves"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve: {e}")
    plt.close()

# --------------------------- 2. Accuracy Curves ---------------------------- #
try:
    plt.figure()
    plt.plot(epochs, mean_train_acc, label="Train (mean)")
    plt.fill_between(
        epochs,
        mean_train_acc - se_train_acc,
        mean_train_acc + se_train_acc,
        alpha=0.3,
        label="Train (±SE)",
    )
    plt.plot(epochs, mean_val_acc, label="Validation (mean)")
    plt.fill_between(
        epochs,
        mean_val_acc - se_val_acc,
        mean_val_acc + se_val_acc,
        alpha=0.3,
        label="Validation (±SE)",
    )
    plt.title("SPR_BENCH Accuracy Curves with Mean ± SE (Sequence Classification)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(_safe_save("mean_accuracy_curves"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy curve: {e}")
    plt.close()

# --------------------------- 3. Macro-F1 Curves ---------------------------- #
try:
    plt.figure()
    plt.plot(epochs, mean_train_f1, label="Train (mean)")
    plt.fill_between(
        epochs,
        mean_train_f1 - se_train_f1,
        mean_train_f1 + se_train_f1,
        alpha=0.3,
        label="Train (±SE)",
    )
    plt.plot(epochs, mean_val_f1, label="Validation (mean)")
    plt.fill_between(
        epochs,
        mean_val_f1 - se_val_f1,
        mean_val_f1 + se_val_f1,
        alpha=0.3,
        label="Validation (±SE)",
    )
    plt.title("SPR_BENCH Macro-F1 Curves with Mean ± SE (Sequence Classification)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.savefig(_safe_save("mean_f1_curves"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated F1 curve: {e}")
    plt.close()

# ---------------------- 4. Aggregated Confusion Matrix --------------------- #
try:
    # Concatenate predictions / gts that are present
    preds_concat = np.concatenate([p for p in test_preds if p.size])
    gts_concat = np.concatenate([g for g in test_gts if g.size])
    if preds_concat.size and gts_concat.size:
        num_classes = int(max(preds_concat.max(), gts_concat.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts_concat, preds_concat):
            cm[gt, pr] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix (Aggregated Test Set)")
        plt.xlabel("Predicted Label")
        plt.ylabel("Ground Truth Label")
        ticks = np.arange(num_classes)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.savefig(_safe_save("aggregated_confusion_matrix"))
        plt.close()

        # Compute per-run accuracy and macro-F1 to report mean±std
        accs, f1s = [], []
        for p, g in zip(test_preds, test_gts):
            if p.size and g.size:
                accs.append((p == g).mean())
                # macro-F1
                nc = int(max(p.max(), g.max()) + 1)
                f1_per_cls = []
                for c in range(nc):
                    tp = ((p == c) & (g == c)).sum()
                    fp = ((p == c) & (g != c)).sum()
                    fn = ((p != c) & (g == c)).sum()
                    prec = tp / (tp + fp) if tp + fp else 0
                    rec = tp / (tp + fn) if tp + fn else 0
                    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
                    f1_per_cls.append(f1)
                f1s.append(np.mean(f1_per_cls))
        if accs:
            accs = np.asarray(accs)
            f1s = np.asarray(f1s)
            print(
                f"Test accuracy: {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}% | "
                f"Test macro-F1: {f1s.mean():.4f} ± {f1s.std():.4f}"
            )
    else:
        print(
            "Predictions or ground-truth not found in any run; skipping confusion matrix."
        )
except Exception as e:
    print(f"Error creating aggregated confusion matrix: {e}")
    plt.close()
