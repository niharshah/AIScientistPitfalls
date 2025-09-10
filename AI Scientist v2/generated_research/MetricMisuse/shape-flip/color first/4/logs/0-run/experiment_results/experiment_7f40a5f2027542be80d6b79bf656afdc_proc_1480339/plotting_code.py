import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = exp["SPR_BENCH"]
except Exception as e:
    print(f"Could not load experiment data: {e}")
    spr = {}


def regroup(entries):
    """entries: list of (budget, epoch, value) -> dict[budget] -> list ordered by epoch"""
    d = {}
    for bud, ep, val in entries:
        d.setdefault(bud, []).append((ep, val))
    for bud in d:
        d[bud] = [v for _, v in sorted(d[bud], key=lambda x: x[0])]
    return d


loss_tr = regroup(spr.get("losses", {}).get("train", []))
loss_val = regroup(spr.get("losses", {}).get("val", []))
cwa_tr = regroup(spr.get("metrics", {}).get("train", []))
cwa_val = regroup(spr.get("metrics", {}).get("val", []))

# --------- plotting (max 5 figures) -------------------------------
plots_made = 0
for bud in sorted(loss_tr)[:5]:  # one plot per bud for loss curves
    try:
        plt.figure()
        plt.plot(loss_tr[bud], label="Train Loss")
        plt.plot(loss_val.get(bud, []), label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Train vs Val Loss ({bud} Epochs)")
        plt.legend()
        fname = f"SPR_BENCH_loss_curves_{bud}epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error plotting loss for {bud}: {e}")
    finally:
        plt.close()
        plots_made += 1

for bud in sorted(cwa_tr)[: 5 - plots_made]:  # remaining plot allowance
    try:
        plt.figure()
        plt.plot(cwa_tr[bud], label="Train CWA")
        plt.plot(cwa_val.get(bud, []), label="Val CWA")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(f"SPR_BENCH Train vs Val CWA ({bud} Epochs)")
        plt.legend()
        fname = f"SPR_BENCH_CWA_curves_{bud}epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error plotting CWA for {bud}: {e}")
    finally:
        plt.close()
        plots_made += 1
    if plots_made >= 4:  # keep last slot for bar chart
        break

# --------- bar chart of best Val CWA per budget -------------------
try:
    best_cwa = {bud: max(vals) for bud, vals in cwa_val.items()}
    if best_cwa:
        plt.figure()
        plt.bar(
            range(len(best_cwa)),
            list(best_cwa.values()),
            tick_label=[str(b) for b in best_cwa.keys()],
        )
        plt.ylabel("Best Validation CWA")
        plt.title("SPR_BENCH Best Val CWA vs Epoch Budget")
        fname = "SPR_BENCH_best_val_CWA_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
    else:
        print("No validation CWA data found.")
except Exception as e:
    print(f"Error plotting bar chart: {e}")
finally:
    plt.close()

# --------- print evaluation metrics -------------------------------
for bud, score in sorted(best_cwa.items()):
    print(f"Epoch budget {bud}: Best Val CWA = {score:.4f}")
