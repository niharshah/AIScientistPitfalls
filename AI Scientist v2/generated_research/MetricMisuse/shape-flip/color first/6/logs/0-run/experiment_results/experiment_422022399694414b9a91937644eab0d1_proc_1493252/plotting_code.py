import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ---------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper to down-sample epochs if needed ---------------------------
def subsample(xs, ys, max_pts=100):
    if len(xs) <= max_pts:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, num=max_pts, dtype=int)
    return [xs[i] for i in idx], [ys[i] for i in idx]


# ---------- plotting ----------------------------------------------------------
test_summary = {}  # dataset -> dict of test metrics
for dset, d in experiment_data.items():
    epochs = d.get("epochs", [])
    # ---------- Loss curve ----------------------------------------------------
    try:
        tr_loss = d["losses"]["train"]
        va_loss = d["losses"]["val"]
        ep_x, tr_loss = subsample(epochs, tr_loss)
        _, va_loss = subsample(epochs, va_loss)
        plt.figure()
        plt.plot(ep_x, tr_loss, label="Train")
        plt.plot(ep_x, va_loss, label="Validation")
        plt.title(f"Loss Curve ({dset})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"loss_curve_{dset}.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
    finally:
        plt.close()

    # ---------- CplxWA curve --------------------------------------------------
    try:
        tr_cplx = d["metrics"]["train_cplxwa"]
        va_cplx = d["metrics"]["val_cplxwa"]
        ep_x, tr_cplx = subsample(epochs, tr_cplx)
        _, va_cplx = subsample(epochs, va_cplx)
        plt.figure()
        plt.plot(ep_x, tr_cplx, label="Train")
        plt.plot(ep_x, va_cplx, label="Validation")
        plt.title(f"CplxWA Curve ({dset})")
        plt.xlabel("Epoch")
        plt.ylabel("CplxWA")
        plt.legend()
        fname = f"cplxwa_curve_{dset}.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating CplxWA curve for {dset}: {e}")
    finally:
        plt.close()

    # ---------- collect test metrics -----------------------------------------
    tmet = {k: v for k, v in d["metrics"].items() if k.startswith("test_")}
    test_summary[dset] = tmet

# ---------- bar plot of test metrics -----------------------------------------
try:
    for metric in ["test_cplxwa", "test_cwa", "test_swa"]:
        if not any(metric in v for v in test_summary.values()):
            continue
        plt.figure()
        names, vals = [], []
        for ds, m in test_summary.items():
            if metric in m:
                names.append(ds)
                vals.append(m[metric])
        plt.bar(names, vals)
        plt.title(f"{metric.replace('_', ' ').title()} Across Datasets")
        plt.ylabel(metric.replace("_", " ").title())
        fname = f"{metric}_summary.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating summary bar plot: {e}")
finally:
    plt.close()

# ---------- print evaluation metrics -----------------------------------------
for ds, m in test_summary.items():
    print(f"\n=== {ds} test metrics ===")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
