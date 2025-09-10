import matplotlib.pyplot as plt
import numpy as np
import os

# ensure working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# navigate to SPR_BENCH sweep results
spr_results = experiment_data.get("epoch_tuning", {}).get("SPR_BENCH", {})

# ------------- print test metrics table -------------
print("\n=== SPR_BENCH test metrics ===")
header = "{:10s} {:>6s} {:>6s} {:>6s} {:>8s}"
print(header.format("run", "ACC", "CWA", "SWA", "CompWA"))
for run, data in spr_results.items():
    tm = data["metrics"]["test"]
    print(
        header.format(
            run,
            f"{tm['acc']:.3f}",
            f"{tm['cwa']:.3f}",
            f"{tm['swa']:.3f}",
            f"{tm['compwa']:.3f}",
        )
    )

# ------------- per-run loss curves -------------
for run, data in spr_results.items():
    try:
        tl = data["losses"]["train"]
        vl = data["losses"]["val"]
        epochs = list(range(1, len(tl) + 1))

        plt.figure()
        plt.plot(epochs, tl, label="Train Loss")
        plt.plot(epochs, vl, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Training vs Validation Loss ({run})")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_loss_{run}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {run}: {e}")
        plt.close()

# ------------- summary metrics vs epochs -------------
try:
    epoch_counts, accs, cwas, swas, compwas = [], [], [], [], []
    for run in sorted(spr_results.keys(), key=lambda x: int(x.split("_")[-1])):
        ep = int(run.split("_")[-1])
        tm = spr_results[run]["metrics"]["test"]
        epoch_counts.append(ep)
        accs.append(tm["acc"])
        cwas.append(tm["cwa"])
        swas.append(tm["swa"])
        compwas.append(tm["compwa"])

    plt.figure()
    for vals, lab in zip([accs, cwas, swas, compwas], ["ACC", "CWA", "SWA", "CompWA"]):
        plt.plot(epoch_counts, vals, marker="o", label=lab)
    plt.xlabel("Training Epochs")
    plt.ylabel("Metric Value")
    plt.title("SPR_BENCH Test Metrics vs Training Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_vs_epochs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error plotting summary metrics: {e}")
    plt.close()
