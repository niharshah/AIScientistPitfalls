import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------- load data ---------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tuning = experiment_data.get("embed_dim_tuning", {})
embed_keys = sorted(tuning.keys())  # e.g. ['embed_64', ...]
dataset_name = "synthetic_SPR"  # underlying dataset


# --------------------- helper ------------------------
def unpack(run_store, path):
    """path like ('losses','train') returns epoch list, value list"""
    items = run_store
    for p in path:
        items = items[p]
    epochs, vals = zip(*items)
    return np.array(epochs), np.array(vals)


plot_count = 0
max_plots = 5

# ------------- 1-3: loss curves per embedding --------
for k in embed_keys:
    if plot_count >= max_plots:
        break
    try:
        run = tuning[k]
        ep_tr, tr_loss = unpack(run, ("losses", "train"))
        ep_va, va_loss = unpack(run, ("losses", "val"))
        plt.figure()
        plt.plot(ep_tr, tr_loss, label="Train")
        plt.plot(ep_va, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f'Loss Curves ({dataset_name})\nEmbedding dim = {k.split("_")[1]}')
        plt.legend()
        fname = f"{dataset_name}_loss_{k}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {k}: {e}")
        plt.close()
    plot_count += 1

# ------------- 4: CoWA vs epoch across dims ----------
if plot_count < max_plots:
    try:
        plt.figure()
        for k in embed_keys:
            ep, cowa = unpack(tuning[k], ("metrics", "CoWA"))
            plt.plot(ep, cowa, label=k.split("_")[1])
        plt.xlabel("Epoch")
        plt.ylabel("CoWA")
        plt.title(
            f"CoWA over Epochs ({dataset_name})\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.legend(title="Embed dim")
        fname = f"{dataset_name}_CoWA_epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CoWA comparison plot: {e}")
        plt.close()
    plot_count += 1

# ------------- 5: final CoWA bar chart ---------------
if plot_count < max_plots:
    try:
        dims, finals = [], []
        for k in embed_keys:
            dims.append(k.split("_")[1])
            finals.append(unpack(tuning[k], ("metrics", "CoWA"))[1][-1])
        x = np.arange(len(dims))
        plt.figure()
        plt.bar(x, finals, color="skyblue")
        plt.xticks(x, dims)
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Final CoWA")
        plt.title(f"Final CoWA by Embedding Size ({dataset_name})")
        fname = f"{dataset_name}_final_CoWA_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating final CoWA bar chart: {e}")
        plt.close()
