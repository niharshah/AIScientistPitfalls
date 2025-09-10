import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    records = experiment_data["multi_dataset_generalization"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    records = {}


# ---------------------------------------------------------------------
# Helper to extract curves
def extract_curve(rec, key):
    # returns epochs, train_vals, val_vals
    tr = rec["losses" if key == "loss" else "metrics"][
        "train" if key in ["loss", "PCWA"] else ""
    ]
    vl = rec["losses" if key == "loss" else "metrics"][
        "val" if key in ["loss", "PCWA"] else ""
    ]
    epochs = [e for e, _ in tr]
    tr_vals = [v for _, v in tr]
    vl_vals = [v for _, v in vl]
    return epochs, tr_vals, vl_vals


# ---------------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    for name, rec in records.items():
        ep, tr, vl = extract_curve(rec, "loss")
        plt.plot(ep, tr, label=f"{name}-train")
        plt.plot(ep, vl, linestyle="--", label=f"{name}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss (All Datasets)")
    plt.legend()
    fname = os.path.join(working_dir, "multi_dataset_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 2) PCWA curves
try:
    plt.figure()
    for name, rec in records.items():
        ep = [e for e, _ in rec["metrics"]["train"]]
        tr = [v for _, v in rec["metrics"]["train"]]
        vl = [v for _, v in rec["metrics"]["val"]]
        plt.plot(ep, tr, label=f"{name}-train")
        plt.plot(ep, vl, linestyle="--", label=f"{name}-val")
    plt.xlabel("Epoch")
    plt.ylabel("PCWA")
    plt.title("Training vs Validation PCWA (All Datasets)")
    plt.legend()
    fname = os.path.join(working_dir, "multi_dataset_pcwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating PCWA plot: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Final test metric summary
try:
    metrics_names = ["ACC", "PCWA", "CWA", "SWA"]
    datasets = list(records.keys())
    bar_width = 0.18
    x = np.arange(len(datasets))
    plt.figure(figsize=(8, 4))
    for i, m in enumerate(metrics_names):
        vals = [records[d]["test_metrics"].get(m, 0.0) for d in datasets]
        plt.bar(x + i * bar_width, vals, width=bar_width, label=m)
    plt.xticks(x + bar_width * 1.5, datasets, rotation=45)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Final Test Metrics by Dataset")
    plt.legend()
    fname = os.path.join(working_dir, "final_test_metrics_summary.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric summary plot: {e}")
    plt.close()
