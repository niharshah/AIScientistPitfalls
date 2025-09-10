import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("embed_dim_tuning", {}).get("SPR_BENCH", {})
embed_dims = sorted([int(k) for k in spr_data.keys()])


# helper: collect curves per key
def collect_curve(key):
    return {dim: spr_data[str(dim)]["metrics"][key] for dim in embed_dims}


loss_train = {d: spr_data[str(d)]["losses"]["train"] for d in embed_dims}
loss_dev = {d: spr_data[str(d)]["losses"]["dev"] for d in embed_dims}
acc_train = collect_curve("train_acc")
acc_dev = collect_curve("dev_acc")

finals = {d: spr_data[str(d)]["final"] for d in embed_dims}

# ------------------- plotting --------------------
# 1) Loss curves
try:
    plt.figure(figsize=(6, 4))
    for d in embed_dims:
        plt.plot(loss_train[d], label=f"train (emb={d})", linestyle="-")
        plt.plot(loss_dev[d], label=f"dev (emb={d})", linestyle="--")
    plt.title("SPR_BENCH: Train vs Dev Loss\nLeft: Solid=Train, Right: Dashed=Dev")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_compare.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# 2) Accuracy curves
try:
    plt.figure(figsize=(6, 4))
    for d in embed_dims:
        plt.plot(acc_train[d], label=f"train (emb={d})", linestyle="-")
        plt.plot(acc_dev[d], label=f"dev (emb={d})", linestyle="--")
    plt.title("SPR_BENCH: Train vs Dev Accuracy\nLeft: Solid=Train, Right: Dashed=Dev")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves_compare.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve figure: {e}")
    plt.close()

# 3) Final accuracy bar chart
try:
    x = np.arange(len(embed_dims))
    width = 0.35
    dev_acc = [finals[d]["dev_acc"] for d in embed_dims]
    test_acc = [finals[d]["test_acc"] for d in embed_dims]

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, dev_acc, width, label="Dev")
    plt.bar(x + width / 2, test_acc, width, label="Test")
    plt.xticks(x, embed_dims)
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Final Accuracy vs Embedding Dim")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_final_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy figure: {e}")
    plt.close()

# 4) RGS, SWA, CWA bar chart
try:
    x = np.arange(len(embed_dims))
    width = 0.2
    dev_rgs = [finals[d]["dev_rgs"] for d in embed_dims]
    test_rgs = [finals[d]["test_rgs"] for d in embed_dims]
    swa = [finals[d]["SWA"] for d in embed_dims]
    cwa = [finals[d]["CWA"] for d in embed_dims]

    plt.figure(figsize=(7, 4))
    plt.bar(x - 1.5 * width, dev_rgs, width, label="Dev RGS")
    plt.bar(x - 0.5 * width, test_rgs, width, label="Test RGS")
    plt.bar(x + 0.5 * width, swa, width, label="SWA")
    plt.bar(x + 1.5 * width, cwa, width, label="CWA")
    plt.xticks(x, embed_dims)
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Robustness & Weighted Accuracies")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_generalisation_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating robustness figure: {e}")
    plt.close()
