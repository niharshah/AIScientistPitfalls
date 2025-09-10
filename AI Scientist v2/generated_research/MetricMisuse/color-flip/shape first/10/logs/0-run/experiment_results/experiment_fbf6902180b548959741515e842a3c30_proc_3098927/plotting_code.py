import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

bench = experiment_data["embedding_dim"]["SPR_BENCH"]

train_loss = bench["losses"]["train"]  # list length 20
val_loss = bench["losses"]["val"]
swa = bench["metrics"]["SWA"]
cwa = bench["metrics"]["CWA"]
hwa = bench["metrics"]["HWA"]
embed_dims = bench["config_values"]  # [32,64,128,256]

# helper for epoch index
epochs = np.arange(1, len(train_loss) + 1)

# 1) Loss curves ---------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Metric curves -------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, hwa, label="HWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Weighted Accuracy Metrics over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_metric_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# 3) Final accuracy per embedding dimension ------------------------------------
try:
    acc = []
    for gt, pr in zip(bench["ground_truth"], bench["predictions"]):
        gt = np.array(gt)
        pr = np.array(pr)
        acc.append((gt == pr).mean())
    x = np.arange(len(embed_dims))
    plt.figure()
    plt.bar(x, acc, tick_label=embed_dims)
    plt.ylim(0, 1)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Final Epoch Accuracy")
    plt.title("SPR_BENCH: Accuracy vs. Embedding Size (Last Epoch)")
    fname = os.path.join(working_dir, "spr_bench_accuracy_by_dim.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
