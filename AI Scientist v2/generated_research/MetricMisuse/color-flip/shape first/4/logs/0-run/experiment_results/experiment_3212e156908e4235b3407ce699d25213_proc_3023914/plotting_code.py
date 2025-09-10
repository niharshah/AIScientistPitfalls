import matplotlib.pyplot as plt
import numpy as np
import os

# Working directory setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----------------- helper extraction -----------------
dataset = "SPR_BENCH"
sweep_key = "embedding_dim_tuning"
sub_exps = experiment_data.get(sweep_key, {}).get(dataset, {})
if not sub_exps:
    print("No experiment data found for plotting.")
    exit()

embed_dims = sorted([int(k.split("_")[-1]) for k in sub_exps.keys()])
loss_train, loss_val, hwa_curves, final_hwa = {}, {}, {}, {}

for dim in embed_dims:
    key = f"emb_{dim}"
    edict = sub_exps[key]
    loss_train[dim] = edict["losses"]["train"]
    loss_val[dim] = edict["losses"]["val"]
    hwa_curves[dim] = edict["metrics"]["val"]
    final_hwa[dim] = hwa_curves[dim][-1]

# ----------------- plot 1: loss curves -----------------
try:
    plt.figure(figsize=(6, 4))
    for dim in embed_dims:
        epochs = range(1, len(loss_train[dim]) + 1)
        plt.plot(epochs, loss_train[dim], label=f"train dim={dim}")
        plt.plot(epochs, loss_val[dim], linestyle="--", label=f"val dim={dim}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset}: Training/Validation Loss vs Epoch")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, f"{dataset}_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ----------------- plot 2: validation HWA curves -----------------
try:
    plt.figure(figsize=(6, 4))
    for dim in embed_dims:
        epochs = range(1, len(hwa_curves[dim]) + 1)
        plt.plot(epochs, hwa_curves[dim], label=f"dim={dim}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title(f"{dataset}: Validation HWA vs Epoch")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, f"{dataset}_metric_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ----------------- plot 3: final HWA bar chart -----------------
try:
    plt.figure(figsize=(5, 4))
    dims = list(final_hwa.keys())
    scores = [final_hwa[d] for d in dims]
    plt.bar([str(d) for d in dims], scores, color="skyblue")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Final HWA")
    plt.title(f"{dataset}: Final Validation HWA by Embedding Size")
    fname = os.path.join(working_dir, f"{dataset}_final_hwa_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# ----------------- print final HWA values -----------------
print("Final validation HWA by embedding dimension:")
for d in embed_dims:
    print(f"  dim={d}: {final_hwa[d]:.3f}")
