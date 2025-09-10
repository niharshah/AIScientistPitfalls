import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load many runs ----------
try:
    experiment_data_path_list = [
        "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_94f1f36a6e4940b4a95927c258449669_proc_3171077/experiment_data.npy",
        "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_8dc7b3be3219466b93f4191cd76ac2fe_proc_3171075/experiment_data.npy",
        "experiments/2025-08-17_00-44-27_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_da70709a9e304e228cd429fab203a01c_proc_3171076/experiment_data.npy",
    ]
    all_experiment_data = []
    root_dir = os.getenv("AI_SCIENTIST_ROOT", os.getcwd())
    for p in experiment_data_path_list:
        full = os.path.join(root_dir, p)
        if os.path.isfile(full):
            all_experiment_data.append(np.load(full, allow_pickle=True).item())
        else:
            print(f"Warning: file not found {full}")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# No data – nothing to do
if not all_experiment_data:
    print("No experiment data found; exiting.")
    quit()

tags = list(all_experiment_data[0].keys())  # assume same keys


def stack_metric(tag, split, metric_key):
    """Return epochs, mean, std_error for given metric across runs."""
    # gather arrays, align by minimum length
    arrays = []
    epochs = None
    for run in all_experiment_data:
        ed = run[tag]
        if split == "losses":
            arr = np.asarray(ed["losses"][metric_key])
        else:
            arr = np.asarray(ed["metrics"][metric_key])
        arrays.append(arr)
        if epochs is None:
            epochs = np.asarray(ed["epochs"])
        else:
            epochs = epochs[: len(arr)]
    min_len = min(map(len, arrays))
    arrays = [a[:min_len] for a in arrays]
    epochs = epochs[:min_len]
    stacked = np.stack(arrays, axis=0)  # runs x epochs
    mean = stacked.mean(axis=0)
    se = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    return epochs, mean, se


# ---------- plot 1: Train / Val Loss with mean ± SE ----------
try:
    plt.figure()
    colors = dict(pretrain_plus_cls="tab:blue", scratch_cls="tab:orange")
    legend_items = []
    for tag, color in zip(["pretrain+cls", "scratch_cls"], colors.values()):
        ep, train_mean, train_se = stack_metric(tag, "losses", "train")
        _, val_mean, val_se = stack_metric(tag, "losses", "val")
        plt.plot(ep, train_mean, linestyle="--", color=color, label=f"{tag} train mean")
        plt.fill_between(
            ep,
            train_mean - train_se,
            train_mean + train_se,
            color=color,
            alpha=0.2,
            label=f"{tag} train ±SE",
        )
        plt.plot(ep, val_mean, linestyle="-", color=color, label=f"{tag} val mean")
        plt.fill_between(
            ep,
            val_mean - val_se,
            val_mean + val_se,
            color=color,
            alpha=0.4,
            label=f"{tag} val ±SE",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss with Mean ± SE (SPR Bench)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sprbench_train_val_loss_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated train/val loss plot: {e}")
    plt.close()

# ---------- plot 2: Train / Val Macro-F1 with mean ± SE ----------
try:
    plt.figure()
    colors = dict(pretrain_plus_cls="tab:green", scratch_cls="tab:red")
    for tag, color in zip(["pretrain+cls", "scratch_cls"], colors.values()):
        ep, train_mean, train_se = stack_metric(tag, "metrics", "train_macro_f1")
        _, val_mean, val_se = stack_metric(tag, "metrics", "val_macro_f1")
        plt.plot(ep, train_mean, linestyle="--", color=color, label=f"{tag} train mean")
        plt.fill_between(
            ep,
            train_mean - train_se,
            train_mean + train_se,
            color=color,
            alpha=0.2,
            label=f"{tag} train ±SE",
        )
        plt.plot(ep, val_mean, linestyle="-", color=color, label=f"{tag} val mean")
        plt.fill_between(
            ep,
            val_mean - val_se,
            val_mean + val_se,
            color=color,
            alpha=0.4,
            label=f"{tag} val ±SE",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Train / Validation Macro-F1 with Mean ± SE (SPR Bench)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sprbench_train_val_macro_f1_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated train/val macro-F1 plot: {e}")
    plt.close()

# ---------- plot 3: Validation Macro-F1 mean comparison ----------
try:
    plt.figure()
    for tag, marker in zip(["pretrain+cls", "scratch_cls"], ["o", "s"]):
        ep, val_mean, val_se = stack_metric(tag, "metrics", "val_macro_f1")
        plt.errorbar(ep, val_mean, yerr=val_se, fmt=f"-{marker}", capsize=3, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Val Macro-F1")
    plt.title("Validation Macro-F1 Mean ± SE Comparison (SPR Bench)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sprbench_val_macro_f1_mean_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val macro-F1 mean comparison plot: {e}")
    plt.close()


# ---------- plot 4: Aggregated Confusion Matrix ----------
def cmatrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


try:
    for tag in ["pretrain+cls", "scratch_cls"]:
        # aggregate matrices
        total_cm = None
        for run in all_experiment_data:
            ed = run[tag]
            y_true = np.asarray(ed.get("ground_truth", []))
            y_pred = np.asarray(ed.get("predictions", []))
            if y_true.size == 0:  # skip if no data
                continue
            cm = cmatrix(y_true, y_pred, num_classes=len(np.unique(y_true)))
            total_cm = cm if total_cm is None else total_cm + cm
        if total_cm is None:
            continue
        avg_cm = total_cm / len(all_experiment_data)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(avg_cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Average Confusion Matrix (SPR Bench) – {tag}")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"sprbench_confmat_avg_{tag}.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating aggregated confusion matrix plot: {e}")
    plt.close()

# ---------- print final metrics ----------
try:
    for tag in ["pretrain+cls", "scratch_cls"]:
        _, val_mean, val_se = stack_metric(tag, "metrics", "val_macro_f1")
        print(f"{tag} final Val Macro-F1: {val_mean[-1]:.4f} ± {val_se[-1]:.4f}")
except Exception as e:
    print(f"Error printing final metrics: {e}")
