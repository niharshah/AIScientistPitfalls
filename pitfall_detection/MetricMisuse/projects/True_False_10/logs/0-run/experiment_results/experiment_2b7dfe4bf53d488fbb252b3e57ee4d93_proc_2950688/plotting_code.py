import matplotlib.pyplot as plt
import numpy as np
import os

# workspace
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# assume toy if real dataset missing
dataset_name = "SPR_BENCH (toy)"

# ------------- 1-3: loss curves for each mode -----------------
for mode, run in experiment_data.items():
    try:
        tr, vl = run["losses"]["train"], run["losses"]["val"]
        ep = np.arange(1, len(tr) + 1)
        plt.figure()
        plt.plot(ep, tr, label="Train")
        plt.plot(ep, vl, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{mode.capitalize()} Loss Curve\nDataset: {dataset_name}")
        plt.legend()
        fname = os.path.join(
            working_dir, f"{dataset_name.replace(' ','_')}_{mode}_loss.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {mode}: {e}")
        plt.close()

# ------------- 4: val SWA progression (≤5 pts) -----------------
try:
    plt.figure()
    for mode, run in experiment_data.items():
        swa = np.array(run["metrics"]["val"])
        if len(swa) > 5:  # sample at most 5 evenly-spaced epochs
            idx = np.round(np.linspace(0, len(swa) - 1, 5)).astype(int)
            swa, ep = swa[idx], idx + 1
        else:
            ep = np.arange(1, len(swa) + 1)
        plt.plot(ep, swa, marker="o", label=mode.capitalize())
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title(f"Validation SWA vs Epoch\nDataset: {dataset_name}")
    plt.legend()
    fname = os.path.join(
        working_dir, f"{dataset_name.replace(' ','_')}_val_SWA_curves.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val SWA plot: {e}")
    plt.close()

# ------------- 5: bar chart of test SWA ------------------------
try:
    modes, test_swa = [], []
    for m, run in experiment_data.items():
        modes.append(m.capitalize())
        test_swa.append(run["test"]["SWA"])
    x = np.arange(len(modes))
    plt.figure(figsize=(6, 4))
    plt.bar(x, test_swa, color="steelblue")
    plt.xticks(x, modes, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title(f"Test SWA Comparison\nDataset: {dataset_name}")
    fname = os.path.join(
        working_dir, f"{dataset_name.replace(' ','_')}_test_SWA_bar.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test SWA bar plot: {e}")
    plt.close()

# ------------- print numeric test metrics ----------------------
print("Final Test Metrics (loss, SWA):")
for mode, run in experiment_data.items():
    tl, ts = run["test"]["loss"], run["test"]["SWA"]
    print(f"{mode.capitalize()}: loss={tl:.4f}, SWA={ts:.4f}")
