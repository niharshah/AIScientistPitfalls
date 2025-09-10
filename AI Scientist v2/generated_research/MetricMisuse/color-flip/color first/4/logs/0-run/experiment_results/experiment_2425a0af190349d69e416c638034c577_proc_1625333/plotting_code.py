import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load data ------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data["epoch_tuning"]["SPR_BENCH"]

    # ---------------- FIGURE 1: loss curves ---------------- #
    try:
        plt.figure(figsize=(7, 4))
        for name, hist in runs.items():
            ep = np.arange(1, len(hist["losses"]["train"]) + 1)
            plt.plot(ep, hist["losses"]["train"], "--", label=f"{name}-train")
            plt.plot(ep, hist["losses"]["val"], "-", label=f"{name}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss Curves (Epoch Tuning)")
        plt.legend()
        fpath = os.path.join(working_dir, "SPR_BENCH_loss_curves_epoch_tuning.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------------- FIGURE 2: validation accuracy -------- #
    try:
        plt.figure(figsize=(7, 4))
        for name, hist in runs.items():
            acc = [d["acc"] for d in hist["metrics"]["val"]]
            ep = np.arange(1, len(acc) + 1)
            plt.plot(ep, acc, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Validation Accuracy Curves (Epoch Tuning)")
        plt.legend()
        plt.ylim(0, 1)
        fpath = os.path.join(working_dir, "SPR_BENCH_val_accuracy_epoch_tuning.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------------- FIGURE 3: validation CWA ------------- #
    try:
        plt.figure(figsize=(7, 4))
        for name, hist in runs.items():
            cwa = [d["cwa"] for d in hist["metrics"]["val"]]
            ep = np.arange(1, len(cwa) + 1)
            plt.plot(ep, cwa, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Color-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation CWA Curves (Epoch Tuning)")
        plt.legend()
        plt.ylim(0, 1)
        fpath = os.path.join(working_dir, "SPR_BENCH_val_cwa_epoch_tuning.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # ---------------- FIGURE 4: final metrics bar ---------- #
    try:
        labels = []
        accs = []
        cwas = []
        swas = []
        pcwas = []
        for name, hist in runs.items():
            final = hist["metrics"]["val"][-1]
            labels.append(name)
            accs.append(final["acc"])
            cwas.append(final["cwa"])
            swas.append(final["swa"])
            pcwas.append(final["pcwa"])

        x = np.arange(len(labels))
        width = 0.2
        plt.figure(figsize=(8, 4))
        plt.bar(x - 1.5 * width, accs, width, label="ACC")
        plt.bar(x - 0.5 * width, cwas, width, label="CWA")
        plt.bar(x + 0.5 * width, swas, width, label="SWA")
        plt.bar(x + 1.5 * width, pcwas, width, label="PCWA")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Final Validation Metrics by Max Epoch Setting")
        plt.legend()
        fpath = os.path.join(working_dir, "SPR_BENCH_final_metrics_epoch_tuning.png")
        plt.savefig(fpath)
        plt.close()
    except Exception as e:
        print(f"Error creating final metrics plot: {e}")
        plt.close()
