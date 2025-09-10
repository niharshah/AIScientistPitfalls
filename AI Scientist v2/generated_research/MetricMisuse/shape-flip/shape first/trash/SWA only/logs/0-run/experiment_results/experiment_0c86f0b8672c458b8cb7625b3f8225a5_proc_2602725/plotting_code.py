import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

sweep = experiment_data["learning_rate_tuning"]["SPR_BENCH"]
lrs = sorted(sweep.keys(), key=lambda k: float(k.split("_")[-1].replace("e", "E")))


# Helpers to fetch a list for each lr
def fetch(path):
    out = []
    for lr in lrs:
        d = sweep[lr]
        cur = d
        for p in path:
            cur = cur[p]
        out.append(cur)
    return out


# ------------------------------------------------------------
# 1) Loss curves
# ------------------------------------------------------------
try:
    plt.figure()
    for lr in lrs:
        tr = sweep[lr]["losses"]["train"]
        vl = sweep[lr]["losses"]["val"]
        plt.plot(tr, label=f"{lr}-train")
        plt.plot(vl, "--", label=f"{lr}-val")
    plt.legend()
    plt.title("Loss Curves – SPR_BENCH (learning-rate sweep)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------
# 2) Accuracy curves
# ------------------------------------------------------------
try:
    plt.figure()
    for lr in lrs:
        tr = sweep[lr]["metrics"]["train"]
        vl = sweep[lr]["metrics"]["val"]
        plt.plot(tr, label=f"{lr}-train")
        plt.plot(vl, "--", label=f"{lr}-val")
    plt.legend()
    plt.title("Accuracy Curves – SPR_BENCH (learning-rate sweep)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# ------------------------------------------------------------
# 3) Final test accuracy bar plot
# ------------------------------------------------------------
try:
    test_accs = [d["metrics"]["test"]["acc"] for d in sweep.values()]
    plt.figure()
    plt.bar(range(len(lrs)), test_accs, tick_label=lrs)
    plt.title("Final Test Accuracy – SPR_BENCH")
    plt.ylabel("Accuracy")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 4) NRGS bar plot
# ------------------------------------------------------------
try:
    nrgs = fetch(["NRGS"])
    plt.figure()
    plt.bar(range(len(lrs)), nrgs, tick_label=lrs)
    plt.title("NRGS – Novel Rule Generalization Score")
    plt.ylabel("NRGS")
    fname = os.path.join(working_dir, "SPR_BENCH_NRGS.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating NRGS plot: {e}")
    plt.close()

# ------------------------------------------------------------
# 5) SWA vs CWA scatter
# ------------------------------------------------------------
try:
    swa = fetch(["metrics", "test", "swa"])
    cwa = fetch(["metrics", "test", "cwa"])
    plt.figure()
    plt.scatter(swa, cwa)
    for i, lr in enumerate(lrs):
        plt.annotate(lr, (swa[i], cwa[i]))
    plt.title("Left: Ground Truth, Right: Generated Samples\nSWA vs CWA – SPR_BENCH")
    plt.xlabel("SWA")
    plt.ylabel("CWA")
    fname = os.path.join(working_dir, "SPR_BENCH_swa_vs_cwa.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA vs CWA plot: {e}")
    plt.close()

# ------------------------------------------------------------
# Print metrics table
# ------------------------------------------------------------
header = f"{'LR':>10} | {'TestAcc':>8} | {'SWA':>6} | {'CWA':>6} | {'NRGS':>6}"
print(header)
print("-" * len(header))
for lr, acc, s, c, n in zip(lrs, test_accs, swa, cwa, nrgs):
    print(f"{lr:>10} | {acc:8.3f} | {s:6.3f} | {c:6.3f} | {n:6.3f}")
