import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("emb_dim", {}).get("SPR_BENCH", {})
dims = sorted(spr.keys(), key=int)  # ['32','64','128','256']

# -------------------- collect arrays --------------------
epochs_dict, tr_loss, val_loss, val_scwa, test_scwa = {}, {}, {}, {}, {}
for d in dims:
    log = spr[d]
    epochs_dict[d] = np.array(log["epochs"])
    tr_loss[d] = np.array(log["losses"]["train"])
    val_loss[d] = np.array(log["losses"]["val"])
    val_scwa[d] = np.array(log["metrics"]["val"])
    test_scwa[d] = log.get("test_SCWA", np.nan)

# -------------------- plot 1: loss curves --------------------
try:
    plt.figure(figsize=(10, 4))
    plt.suptitle("SPR_BENCH Loss Curves\nLeft: Training Loss, Right: Validation Loss")
    plt.subplot(1, 2, 1)
    for d in dims:
        plt.plot(epochs_dict[d], tr_loss[d], label=f"emb={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    for d in dims:
        plt.plot(epochs_dict[d], val_loss[d], label=f"emb={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    fp = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fp)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# -------------------- plot 2: validation SCWA --------------------
try:
    plt.figure()
    for d in dims:
        plt.plot(epochs_dict[d], val_scwa[d], marker="o", label=f"emb={d}")
    plt.title("SPR_BENCH Validation SCWA over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SCWA")
    plt.legend()
    fp = os.path.join(working_dir, "SPR_BENCH_val_SCWA_curves.png")
    plt.savefig(fp)
    plt.close()
except Exception as e:
    print(f"Error creating val SCWA plot: {e}")
    plt.close()

# -------------------- plot 3: test SCWA bar --------------------
try:
    plt.figure()
    xs = np.arange(len(dims))
    plt.bar(xs, [test_scwa[d] for d in dims])
    plt.xticks(xs, dims)
    plt.ylabel("Test SCWA")
    plt.title("SPR_BENCH Test SCWA by Embedding Dimension")
    fp = os.path.join(working_dir, "SPR_BENCH_test_SCWA_bar.png")
    plt.savefig(fp)
    plt.close()
except Exception as e:
    print(f"Error creating test SCWA bar plot: {e}")
    plt.close()

# -------------------- print evaluation table --------------------
print("Embedding Dim -> Test SCWA")
for d in dims:
    print(f"{d:>3}: {test_scwa[d]:.4f}")
