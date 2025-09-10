import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    edict = experiment_data.get("embedding_dim_tuning", {})
    emb_keys = sorted(edict.keys(), key=lambda k: int(k.split("emb")[-1]))
    epochs = range(1, len(next(iter(edict.values()))["losses"]["train"]) + 1)

    # gather test metrics for printing
    tests = {}
    for k in emb_keys:
        tests[int(k.split("emb")[-1])] = edict[k]["metrics"]["test"]

    # ------------- Figure 1: loss curves -------------
    try:
        plt.figure(figsize=(8, 4))
        # left subplot: train loss
        plt.subplot(1, 2, 1)
        for k in emb_keys:
            plt.plot(epochs, edict[k]["losses"]["train"], label=k)
        plt.title("Train Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize=6)

        # right subplot: val loss
        plt.subplot(1, 2, 2)
        for k in emb_keys:
            plt.plot(epochs, edict[k]["losses"]["val"], label=k)
        plt.title("Validation Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize=6)

        plt.suptitle("SPR_BENCH: Loss Curves Across Embedding Dimensions")
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve figure: {e}")
        plt.close()

    # ------------- Figure 2: validation HWA curves -------------
    try:
        plt.figure()
        for k in emb_keys:
            hwa_vals = [m["HWA"] for m in edict[k]["metrics"]["val"]]
            plt.plot(epochs, hwa_vals, label=k)
        plt.title("Validation HWA vs Epochs (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend(fontsize=6)
        fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve figure: {e}")
        plt.close()

    # ------------- Figure 3: final test metric bars -------------
    try:
        labels = ["SWA", "CWA", "HWA"]
        x = np.arange(len(emb_keys))
        width = 0.25
        plt.figure()
        for i, metric in enumerate(labels):
            vals = [tests[int(k.split("emb")[-1])][metric] for k in emb_keys]
            plt.bar(x + i * width - width, vals, width, label=metric)
        plt.xticks(x, [k for k in emb_keys])
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Test Metrics by Embedding Dim")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics figure: {e}")
        plt.close()

    # ------------- print test metric table -------------
    print("Final Test Metrics:")
    for emb_dim, metrics in sorted(tests.items()):
        print(
            f"  emb={emb_dim}:  SWA={metrics['SWA']:.4f}  CWA={metrics['CWA']:.4f}  HWA={metrics['HWA']:.4f}"
        )
