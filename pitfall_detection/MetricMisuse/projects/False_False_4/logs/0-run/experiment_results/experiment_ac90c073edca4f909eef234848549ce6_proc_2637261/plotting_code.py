import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

saved = []
if spr:
    # convenience variables
    train_loss = spr["losses"]["train"]
    dev_loss = spr["losses"]["dev"]
    train_acc = [d["ACC"] for d in spr["metrics"]["train"]]
    dev_acc = [d["ACC"] for d in spr["metrics"]["dev"]]
    dev_metrics = spr["metrics"]["dev"]
    epochs = np.arange(1, len(train_loss) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, dev_loss, label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.tight_layout()
        plt.savefig(fname)
        saved.append(fname)
    except Exception as e:
        print(f"Error creating loss plot: {e}")
    finally:
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, dev_acc, label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Accuracy Curves")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_accuracy_curve.png")
        plt.tight_layout()
        plt.savefig(fname)
        saved.append(fname)
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
    finally:
        plt.close()

    # 3) Dev-set SWA / CWA / WGMA
    try:
        swa = [d["SWA"] for d in dev_metrics]
        cwa = [d["CWA"] for d in dev_metrics]
        wgma = [d["WGMA"] for d in dev_metrics]
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, wgma, label="WGMA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Dev Specialty Metrics")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_dev_special_metrics.png")
        plt.tight_layout()
        plt.savefig(fname)
        saved.append(fname)
    except Exception as e:
        print(f"Error creating specialty metrics plot: {e}")
    finally:
        plt.close()

    # 4) Scatter WGMA vs RGS
    try:
        rgs = [d["RGS"] for d in dev_metrics]
        plt.figure()
        plt.scatter(wgma, rgs, c=epochs, cmap="viridis")
        plt.colorbar(label="Epoch")
        plt.xlabel("WGMA")
        plt.ylabel("RGS")
        plt.title("SPR_BENCH – WGMA vs. RGS\n(Colour indicates epoch)")
        fname = os.path.join(working_dir, "spr_bench_wgma_vs_rgs.png")
        plt.tight_layout()
        plt.savefig(fname)
        saved.append(fname)
    except Exception as e:
        print(f"Error creating WGMA-RGS scatter: {e}")
    finally:
        plt.close()

    # 5) Final test metrics bar
    try:
        test_m = spr["metrics"]["test"]
        keys, vals = zip(*[(k, v) for k, v in test_m.items() if k != "loss"])
        plt.figure()
        plt.bar(keys, vals)
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH – Final Test Metrics")
        fname = os.path.join(working_dir, "spr_bench_test_metrics_bar.png")
        plt.tight_layout()
        plt.savefig(fname)
        saved.append(fname)
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
    finally:
        plt.close()

# List saved files
for f in saved:
    print(f"Saved: {f}")
