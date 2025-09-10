import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def get_runs():
    runs = []
    for bi_flag, run in (
        experiment_data.get("bidirectional", {}).get("SPR_BENCH", {}).items()
    ):
        runs.append((bi_flag, run))
    return runs


runs = get_runs()

# ---------------- loss curves ----------------
for bi_flag, run in runs:
    try:
        plt.figure()
        plt.plot(run["losses"]["train"], label="train")
        plt.plot(run["losses"]["dev"], label="validation")
        plt.title(f"SPR_BENCH Loss Curve (bidirectional={bi_flag})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"loss_curve_SPR_BENCH_bi_{bi_flag}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot (bi={bi_flag}): {e}")
        plt.close()

# ---------------- dev metric curves ----------------
metrics = ["acc", "swa", "cwa"]
for bi_flag, run in runs:
    try:
        epochs = list(range(1, len(run["metrics"]["dev"]) + 1))
        plt.figure(figsize=(7, 4))
        for m in metrics:
            vals = [ep[m] for ep in run["metrics"]["dev"]]
            plt.plot(epochs, vals, label=m.upper())
        plt.title(f"SPR_BENCH Dev Metrics (bidirectional={bi_flag})")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend()
        fname = f"dev_metrics_SPR_BENCH_bi_{bi_flag}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating dev-metric plot (bi={bi_flag}): {e}")
        plt.close()

# ---------------- final test comparison ----------------
try:
    labels = ["ACC", "SWA", "CWA", "NRGS"]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure()
    for i, (bi_flag, run) in enumerate(runs):
        test = run["metrics"]["test"]
        vals = [test["acc"], test["swa"], test["cwa"], run["metrics"]["NRGS"]]
        plt.bar(x + i * width, vals, width, label=f"bi={bi_flag}")
    plt.xticks(x + width / 2, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("SPR_BENCH Final Test Metrics")
    plt.legend()
    fname = "test_metrics_comparison_SPR_BENCH.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating test comparison plot: {e}")
    plt.close()

# ---------------- print quick summary ----------------
for bi_flag, run in runs:
    t = run["metrics"]["test"]
    print(
        f"bi={bi_flag} | ACC={t['acc']:.3f} | SWA={t['swa']:.3f} | CWA={t['cwa']:.3f} | NRGS={run['metrics']['NRGS']:.3f}"
    )
