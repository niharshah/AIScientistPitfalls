import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_key = next(iter(experiment_data.keys())) if experiment_data else None
if exp_key is None:
    print("No experiment data to plot.")
    quit()

runs = experiment_data[exp_key]  # e.g. "MultiSynthetic"
epochs = len(next(iter(runs.values()))["losses"]["train"])


# ---------- gather helpers ----------
def get_curve(runs, field):
    curves = {}
    for train_ds, rec in runs.items():
        curves[train_ds] = rec[field]
    return curves


loss_train = get_curve(runs, ("losses", "train"))
loss_val = get_curve(runs, ("losses", "val"))
hwa_val = get_curve(runs, ("metrics", "val"))


def nested_get(dic, keys):
    d = dic
    for k in keys:
        d = d[k]
    return d


# ---------- 1) loss curves ----------
try:
    plt.figure()
    x = np.arange(1, epochs + 1)
    for ds in runs:
        plt.plot(
            x, nested_get(runs[ds], ("losses", "train")), "--", label=f"{ds}-train"
        )
        plt.plot(x, nested_get(runs[ds], ("losses", "val")), "-", label=f"{ds}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{exp_key}: Train vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{exp_key}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2) validation HWA curves ----------
try:
    plt.figure()
    for ds in runs:
        plt.plot(x, nested_get(runs[ds], ("metrics", "val")), label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title(f"{exp_key}: Validation HWA Across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, f"{exp_key}_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# ---------- 3) cross-domain HWA heatmap ----------
try:
    train_names = list(runs.keys())
    eval_names = list(train_names)
    heat = np.zeros((len(train_names), len(eval_names)))
    for r, tr_ds in enumerate(train_names):
        for c, ev_ds in enumerate(eval_names):
            heat[r, c] = runs[tr_ds]["cross_eval"][ev_ds]["hwa"]
    plt.figure()
    im = plt.imshow(heat, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(eval_names)), eval_names, rotation=45, ha="right")
    plt.yticks(range(len(train_names)), train_names)
    plt.title(f"{exp_key}: Cross-Domain HWA (rows=train, cols=eval)")
    for r in range(len(train_names)):
        for c in range(len(eval_names)):
            plt.text(
                c,
                r,
                f"{heat[r,c]:.2f}",
                ha="center",
                va="center",
                color="white" if heat[r, c] < 0.5 else "black",
            )
    fname = os.path.join(working_dir, f"{exp_key}_cross_domain_HWA.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating cross-domain heatmap: {e}")
    plt.close()

# ---------- print metrics ----------
print("\nCross-Domain HWA (train -> eval):")
for tr_ds in runs:
    row = []
    for ev_ds in runs:
        h = runs[tr_ds]["cross_eval"][ev_ds]["hwa"]
        row.append(f"{h:.3f}")
    print(f"{tr_ds:>8}:", "  ".join(row))
