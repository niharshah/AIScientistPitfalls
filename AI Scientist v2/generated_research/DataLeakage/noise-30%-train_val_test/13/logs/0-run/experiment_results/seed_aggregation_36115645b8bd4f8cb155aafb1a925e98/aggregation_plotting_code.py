import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score

# ----------------- basic set-up -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# paths provided in the prompt
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_7b4df9fbec364550a6f9cd010d46ea10_proc_3464963/experiment_data.npy",
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_510b055ead6f4748869474fa247f6a6f_proc_3464962/experiment_data.npy",
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_f28d2d1233934773935c04c13333106d_proc_3464964/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# ----------------- aggregate -----------------
agg = {}  # {nl_key: dict of lists}
for exp in all_experiment_data:
    runs = exp.get("num_layers", {}).get("SPR_BENCH", {})
    for nl_key, data in runs.items():
        entry = agg.setdefault(
            nl_key,
            {
                "epochs": np.asarray(data["epochs"]),
                "train_f1": [],
                "val_f1": [],
                "train_loss": [],
                "val_loss": [],
                "test_f1": [],
            },
        )
        entry["train_f1"].append(np.asarray(data["metrics"]["train_f1"]))
        entry["val_f1"].append(np.asarray(data["metrics"]["val_f1"]))
        entry["train_loss"].append(np.asarray(data["losses"]["train"]))
        entry["val_loss"].append(np.asarray(data["losses"]["val"]))

        # test F1 for this run
        preds = np.asarray(data.get("predictions", []))
        gts = np.asarray(data.get("ground_truth", []))
        if preds.size and gts.size:
            entry["test_f1"].append(f1_score(gts, preds, average="macro"))


# helper to get mean & se
def mean_se(stack):
    arr = np.vstack(stack)
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return mean, se


sorted_keys = sorted(agg, key=lambda x: int(x.split("_")[1]))

# ----------------- plot 1: mean ± se F1 curves -----------------
try:
    plt.figure()
    for nl_key in sorted_keys:
        d = agg[nl_key]
        epochs = d["epochs"]
        if not d["train_f1"] or not d["val_f1"]:
            continue
        m_tr, se_tr = mean_se(d["train_f1"])
        m_val, se_val = mean_se(d["val_f1"])

        plt.plot(epochs, m_tr, label=f"train {nl_key}")
        plt.fill_between(epochs, m_tr - se_tr, m_tr + se_tr, alpha=0.2)
        plt.plot(epochs, m_val, linestyle="--", label=f"val {nl_key}")
        plt.fill_between(epochs, m_val - se_val, m_val + se_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Mean ± SE Training/Validation F1")
    plt.legend(title="Shaded: ±SE")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated F1 plot: {e}")
    plt.close()

# ----------------- plot 2: mean ± se Loss curves -----------------
try:
    plt.figure()
    for nl_key in sorted_keys:
        d = agg[nl_key]
        epochs = d["epochs"]
        if not d["train_loss"] or not d["val_loss"]:
            continue
        m_tr, se_tr = mean_se(d["train_loss"])
        m_val, se_val = mean_se(d["val_loss"])

        plt.plot(epochs, m_tr, label=f"train {nl_key}")
        plt.fill_between(epochs, m_tr - se_tr, m_tr + se_tr, alpha=0.2)
        plt.plot(epochs, m_val, linestyle="--", label=f"val {nl_key}")
        plt.fill_between(epochs, m_val - se_val, m_val + se_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Mean ± SE Training/Validation Loss")
    plt.legend(title="Shaded: ±SE")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ----------------- plot 3: test F1 bar with error -----------------
try:
    plt.figure()
    means, ses = [], []
    for nl_key in sorted_keys:
        scores = agg[nl_key]["test_f1"]
        if scores:
            means.append(np.mean(scores))
            ses.append(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        else:
            means.append(0.0)
            ses.append(0.0)
    x = np.arange(len(sorted_keys))
    plt.bar(x, means, yerr=ses, capsize=5, color="skyblue")
    plt.xticks(x, sorted_keys, rotation=45)
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Test Macro-F1 (Mean ± SE) by num_layers")
    for i, v in enumerate(means):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_agg_test_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated test F1 bar plot: {e}")
    plt.close()

# --------------- print summary ---------------
print("Aggregated Test Macro-F1 (mean ± SE):")
for k, m, s in zip(sorted_keys, means, ses):
    print(f"{k}: {m:.4f} ± {s:.4f}")
