import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- Load experiment results ----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    runs = experiment_data.get("pos_weight", {})
    keys = sorted(runs.keys(), key=lambda k: int(k.split("pw")[-1]))  # pw1, pw2, ...
    epochs = len(next(iter(runs.values()))["losses"]["train"])

    # ----------------------- Figure 1 ------------------------------
    try:
        plt.figure(figsize=(10, 5))
        for k in keys:
            plt.plot(
                range(1, epochs + 1), runs[k]["losses"]["train"], label=f"{k}_train"
            )
            plt.plot(
                range(1, epochs + 1),
                runs[k]["losses"]["val"],
                linestyle="--",
                label=f"{k}_val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------------------- Figure 2 ------------------------------
    try:
        plt.figure(figsize=(8, 5))
        for k in keys:
            comp_wa = runs[k]["metrics"]["val_CompWA"]
            plt.plot(range(1, epochs + 1), comp_wa, label=k)
        plt.xlabel("Epoch")
        plt.ylabel("Validation CompWA")
        plt.title("SPR_BENCH Validation Complexity-Weighted Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_CompWA.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot: {e}")
        plt.close()

    # ----------------------- Figure 3 ------------------------------
    try:
        cwa = [runs[k]["val_CWA"] for k in keys]
        swa = [runs[k]["val_SWA"] for k in keys]
        x = np.arange(len(keys))
        width = 0.35
        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, cwa, width, label="CWA")
        plt.bar(x + width / 2, swa, width, label="SWA")
        plt.xticks(x, keys)
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Final Weighted Accuracies\nLeft: CWA, Right: SWA")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_final_weighted_accs.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final accuracy plot: {e}")
        plt.close()
