import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper to unwrap structure ----------
def extract_runs(ds_blob):
    "Return list of (run_key, record_dict)"
    # ds_blob may already be a run dict (contains 'metrics')
    if "metrics" in ds_blob and "losses" in ds_blob:
        return (
            list(ds_blob.items())
            if isinstance(ds_blob["metrics"], dict)
            else [("", ds_blob)]
        )
    return list(ds_blob.items())


datasets = {}
for top_k, top_v in experiment_data.items():
    if "metrics" in top_v and "losses" in top_v:  # single dataset, top_k == run
        datasets.setdefault("SPR_BENCH", {})[top_k] = top_v
    else:  # multi-dataset
        datasets[top_k] = top_v

# ---------- colours ----------
palette = plt.cm.tab10.colors

# ---------- 1) train / val F1 curves ----------
try:
    for ds_name, runs in datasets.items():
        plt.figure()
        for idx, (run_k, rec) in enumerate(runs.items()):
            c = palette[idx % len(palette)]
            plt.plot(
                rec["epochs"],
                rec["metrics"]["train_macro_f1"],
                linestyle="--",
                color=c,
                label=f"{run_k}-train",
            )
            plt.plot(
                rec["epochs"],
                rec["metrics"]["val_macro_f1"],
                linestyle="-",
                color=c,
                label=f"{run_k}-val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{ds_name} Macro-F1 Curves (Left: Train dashed, Right: Val solid)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name.lower()}_macro_f1_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# ---------- 2) train / val Loss curves ----------
try:
    for ds_name, runs in datasets.items():
        plt.figure()
        for idx, (run_k, rec) in enumerate(runs.items()):
            c = palette[idx % len(palette)]
            plt.plot(
                rec["epochs"],
                rec["losses"]["train"],
                "--",
                color=c,
                label=f"{run_k}-train",
            )
            plt.plot(
                rec["epochs"], rec["losses"]["val"], "-", color=c, label=f"{run_k}-val"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Loss Curves (Left: Train dashed, Right: Val solid)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name.lower()}_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- 3) test Macro-F1 per run ----------
try:
    for ds_name, runs in datasets.items():
        scores = {rk: rec.get("test_macro_f1", np.nan) for rk, rec in runs.items()}
        plt.figure()
        plt.bar(
            range(len(scores)), list(scores.values()), tick_label=list(scores.keys())
        )
        plt.ylabel("Macro-F1")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.title(f"{ds_name} Test Macro-F1 per Run")
        plt.savefig(
            os.path.join(working_dir, f"{ds_name.lower()}_test_macro_f1_bar.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating test bar chart: {e}")
    plt.close()

# ---------- 4) best-per-dataset comparison (if >1 dataset) ----------
try:
    if len(datasets) > 1:
        best_scores = {
            ds: max(r["test_macro_f1"] for r in runs.values())
            for ds, runs in datasets.items()
        }
        plt.figure()
        plt.bar(
            range(len(best_scores)),
            list(best_scores.values()),
            tick_label=list(best_scores.keys()),
        )
        plt.ylabel("Macro-F1")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.title("Best Test Macro-F1 Across Datasets")
        plt.savefig(os.path.join(working_dir, "datasets_best_test_macro_f1.png"))
        plt.close()
except Exception as e:
    print(f"Error creating dataset comparison plot: {e}")
    plt.close()

# ---------- 5) confusion-matrix heatmap for first ds/run ----------
try:
    first_ds = next(iter(datasets))
    first_run = next(iter(datasets[first_ds]))
    rec = datasets[first_ds][first_run]
    preds, trues = np.array(rec["predictions"]), np.array(rec["ground_truth"])
    num_cls = len(np.unique(np.concatenate([preds, trues])))
    conf = np.zeros((num_cls, num_cls), int)
    for t, p in zip(trues, preds):
        conf[t, p] += 1
    plt.figure()
    plt.imshow(conf, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{first_ds} Confusion Matrix ({first_run})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{first_ds.lower()}_{first_run}_conf_mat.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print summary ----------
all_scores = {
    f"{ds}:{rk}": rv.get("test_macro_f1", np.nan)
    for ds, runs in datasets.items()
    for rk, rv in runs.items()
}
print("Test Macro-F1 scores:", all_scores)
