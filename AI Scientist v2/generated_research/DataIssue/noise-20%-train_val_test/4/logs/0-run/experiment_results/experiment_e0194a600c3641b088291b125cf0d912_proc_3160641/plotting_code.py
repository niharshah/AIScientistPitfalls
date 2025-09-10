import matplotlib.pyplot as plt
import numpy as np
import os

# set working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
experiment_path_candidates = [
    os.path.join(working_dir, "experiment_data.npy"),
    os.path.join(os.getcwd(), "experiment_data.npy"),
]
experiment_data = None
for p in experiment_path_candidates:
    try:
        experiment_data = np.load(p, allow_pickle=True).item()
        break
    except Exception:
        pass
if experiment_data is None:
    raise FileNotFoundError("experiment_data.npy not found in expected locations.")

results = experiment_data["nhead_tuning"]["SPR_BENCH"]["results"]

# ---------- per-nhead plots ----------
for nhead, res in results.items():
    hist = res["history"]
    epochs = hist["epochs"]
    # Loss curves
    try:
        plt.figure()
        plt.plot(epochs, hist["losses"]["train"], label="Train Loss")
        plt.plot(epochs, hist["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"SPR_BENCH Loss Curves (nhead={nhead})\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = f"SPR_BENCH_nhead{nhead}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for nhead={nhead}: {e}")
        plt.close()
    # F1 curves
    try:
        plt.figure()
        plt.plot(epochs, hist["metrics"]["train_f1"], label="Train F1")
        plt.plot(epochs, hist["metrics"]["val_f1"], label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(
            f"SPR_BENCH F1 Curves (nhead={nhead})\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = f"SPR_BENCH_nhead{nhead}_f1_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting F1 curves for nhead={nhead}: {e}")
        plt.close()

# ---------- summary bar chart ----------
try:
    plt.figure()
    heads = sorted(results.keys())
    test_f1s = [results[h]["test_f1"] for h in heads]
    plt.bar([str(h) for h in heads], test_f1s, color="skyblue")
    plt.xlabel("nhead")
    plt.ylabel("Test Macro F1")
    plt.title("SPR_BENCH Test F1 by nhead\nLeft: nhead value, Right: Macro F1")
    for i, v in enumerate(test_f1s):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center")
    fname = "SPR_BENCH_test_f1_summary.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error plotting summary bar chart: {e}")
    plt.close()
