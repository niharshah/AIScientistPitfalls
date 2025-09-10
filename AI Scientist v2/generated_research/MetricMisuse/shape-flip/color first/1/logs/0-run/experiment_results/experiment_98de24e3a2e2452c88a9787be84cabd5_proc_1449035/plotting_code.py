import matplotlib.pyplot as plt
import numpy as np
import os

# prepare paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "MultiSynthetic"
rules = list(experiment_data.get(ds_name, {}).keys())

# ---------------- fig 1: loss curves -------------------
try:
    plt.figure(figsize=(6, 4))
    for r in rules:
        tr = experiment_data[ds_name][r]["losses"]["train"]
        vl = experiment_data[ds_name][r]["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, "--", label=f"{r}-train")
        plt.plot(epochs, vl, "-", label=f"{r}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MultiSynthetic: Training vs Validation Loss")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "MultiSynthetic_loss_curves.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------- fig 2: HPA curves --------------------
try:
    plt.figure(figsize=(6, 4))
    for r in rules:
        hpa = [m["HPA"] for m in experiment_data[ds_name][r]["metrics"]["val"]]
        plt.plot(range(1, len(hpa) + 1), hpa, label=r)
    plt.xlabel("Epoch")
    plt.ylabel("HPA")
    plt.title("MultiSynthetic: Validation Harmonic Poly Accuracy")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "MultiSynthetic_HPA_curves.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HPA curve plot: {e}")
    plt.close()

# ---------------- fig 3: cross-task HPA heat-map -------
try:
    hpa_mat = np.zeros((len(rules), len(rules)))
    for i, src in enumerate(rules):
        for j, tgt in enumerate(rules):
            hpa_mat[i, j] = experiment_data[ds_name][src]["cross_test"][tgt]["HPA"]
    plt.figure(figsize=(5, 4))
    im = plt.imshow(hpa_mat, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(rules)), rules, rotation=45, ha="right")
    plt.yticks(range(len(rules)), rules)
    plt.title("MultiSynthetic: Cross-Task HPA\nRows: train rule, Cols: test rule")
    for i in range(len(rules)):
        for j in range(len(rules)):
            plt.text(
                j,
                i,
                f"{hpa_mat[i,j]:.2f}",
                ha="center",
                va="center",
                color="w" if hpa_mat[i, j] < 0.5 else "k",
                fontsize=7,
            )
    fname = os.path.join(working_dir, "MultiSynthetic_cross_task_HPA_heatmap.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating cross-task heatmap: {e}")
    plt.close()

# ---------------- fig 4: CWA vs SWA bars ---------------
try:
    ind = np.arange(len(rules))
    width = 0.35
    cwa = [experiment_data[ds_name][r]["cross_test"][r]["CWA"] for r in rules]
    swa = [experiment_data[ds_name][r]["cross_test"][r]["SWA"] for r in rules]
    plt.figure(figsize=(6, 4))
    plt.bar(ind - width / 2, cwa, width, label="CWA")
    plt.bar(ind + width / 2, swa, width, label="SWA")
    plt.xticks(ind, rules, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("MultiSynthetic: In-Domain Color vs Shape Weighted Acc")
    plt.legend()
    fname = os.path.join(working_dir, "MultiSynthetic_CWA_SWA_bars.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA/SWA bar plot: {e}")
    plt.close()
