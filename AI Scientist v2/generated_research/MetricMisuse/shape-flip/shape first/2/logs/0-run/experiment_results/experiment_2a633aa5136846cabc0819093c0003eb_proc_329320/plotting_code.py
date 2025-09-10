import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helpers ----------
def get_histories(exp_dict):
    rates, losses_tr, losses_dev, pha_tr, pha_dev = [], [], [], [], []
    for k, v in exp_dict.items():
        if k == "best":  # skip special record
            continue
        rates.append(float(k))
        losses_tr.append(v["losses"]["train"])
        losses_dev.append(v["losses"]["dev"])
        pha_tr.append(v["metrics"]["train_PHA"])
        pha_dev.append(v["metrics"]["dev_PHA"])
    # sort by rate
    idx = np.argsort(rates)
    rates = np.asarray(rates)[idx]
    losses_tr = [losses_tr[i] for i in idx]
    losses_dev = [losses_dev[i] for i in idx]
    pha_tr = [pha_tr[i] for i in idx]
    pha_dev = [pha_dev[i] for i in idx]
    return rates, losses_tr, losses_dev, pha_tr, pha_dev


spr_dict = experiment_data.get("dropout_rate", {}).get("spr_bench", {})

rates, loss_tr, loss_dev, pha_tr, pha_dev = get_histories(spr_dict)

# ---------- 1: loss curves ----------
try:
    plt.figure()
    for r, lt, ld in zip(rates, loss_tr, loss_dev):
        epochs = range(1, len(lt) + 1)
        plt.plot(epochs, lt, "--", label=f"dr={r:.1f} train")
        plt.plot(epochs, ld, "-", label=f"dr={r:.1f} dev")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(
        "spr_bench: Training vs Validation Loss\nLeft: Train Dashed, Right: Dev Solid"
    )
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2: PHA curves ----------
try:
    plt.figure()
    for r, pt, pd in zip(rates, pha_tr, pha_dev):
        epochs = range(1, len(pt) + 1)
        plt.plot(epochs, pt, "--", label=f"dr={r:.1f} train")
        plt.plot(epochs, pd, "-", label=f"dr={r:.1f} dev")
    plt.xlabel("Epoch")
    plt.ylabel("PHA")
    plt.title(
        "spr_bench: Training vs Validation PHA\nLeft: Train Dashed, Right: Dev Solid"
    )
    plt.legend(fontsize=6)
    plt.ylim(0, 1)
    fname = os.path.join(working_dir, "spr_bench_pha_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating PHA plot: {e}")
    plt.close()

# ---------- 3: Best dev PHA vs dropout rate ----------
try:
    best_dev = [max(p) for p in pha_dev]
    plt.figure()
    plt.bar([str(r) for r in rates], best_dev, color="skyblue")
    plt.xlabel("Dropout rate")
    plt.ylabel("Best Dev PHA")
    plt.title("spr_bench: Best Dev PHA per Dropout Rate")
    fname = os.path.join(working_dir, "spr_bench_best_dev_PHA_vs_dropout.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating best-dev PHA bar: {e}")
    plt.close()

# ---------- 4: Test metrics of best model ----------
try:
    best_block = spr_dict.get("best", {})
    test_metrics = best_block.get("test_metrics", {})
    if test_metrics:
        names = list(test_metrics.keys())
        vals = [test_metrics[n] for n in names]
        plt.figure()
        plt.bar(names, vals, color=["orange", "green", "purple"])
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title(
            "spr_bench: Best Model Test Metrics\nLeft: SWA, Middle: CWA, Right: PHA"
        )
        fname = os.path.join(working_dir, "spr_bench_best_model_test_metrics.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        print("No test metrics found for best model.")
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

# ---------- print summary ----------
if best_block.get("test_metrics"):
    print("Best model test metrics:", best_block["test_metrics"])
