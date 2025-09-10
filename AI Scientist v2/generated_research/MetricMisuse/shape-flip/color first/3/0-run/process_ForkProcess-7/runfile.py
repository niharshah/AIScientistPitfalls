import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = exp["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp, grid_keys = {}, []
else:
    grid_keys = list(exp.keys())


# ---------------------- helper utilities -----------------------
def best_lr_key(exp_dict):
    best_key, best_val = None, -1
    for k, rec in exp_dict.items():
        val_hist = rec["metrics"]["val"]
        if val_hist and val_hist[-1] > best_val:
            best_val = val_hist[-1]
            best_key = k
    return best_key


best_key = best_lr_key(exp)

# ---------------------------- FIG 1 -----------------------------
try:
    plt.figure()
    for k in grid_keys[:]:
        epochs = range(1, len(exp[k]["metrics"]["val"]) + 1)
        plt.plot(epochs, exp[k]["metrics"]["val"], label=k.replace("lr_", "lr="))
    plt.xlabel("Epoch")
    plt.ylabel("Dev BWA")
    plt.title(
        "Learning-rate comparison on SPR_BENCH\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.legend()
    path = os.path.join(working_dir, "spr_bench_bwa_curves.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# ---------------------------- FIG 2 -----------------------------
try:
    plt.figure()
    for k in grid_keys[:]:
        ep = range(1, len(exp[k]["losses"]["train"]) + 1)
        plt.plot(ep, exp[k]["losses"]["train"], linestyle="-", label=f"{k}_train")
        plt.plot(ep, exp[k]["losses"]["val"], linestyle="--", label=f"{k}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Train/Val Loss for each learning-rate (SPR_BENCH)")
    plt.legend(fontsize=6)
    path = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# ---------------------------- FIG 3 -----------------------------
try:
    if best_key:
        plt.figure()
        ep = range(1, len(exp[best_key]["metrics"]["train"]) + 1)
        plt.plot(ep, exp[best_key]["metrics"]["train"], label="Train BWA")
        plt.plot(ep, exp[best_key]["metrics"]["val"], label="Dev BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"Best lr ({best_key.replace('lr_','')}) – Train vs Dev BWA")
        plt.legend()
        path = os.path.join(working_dir, f"spr_bench_best_lr_bwa.png")
        plt.savefig(path)
        print(f"Saved {path}")
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# ---------------------------- FIG 4 -----------------------------
try:
    plt.figure()
    final_bwa = [exp[k]["test_bwa"] for k in grid_keys]
    lrs = [k.replace("lr_", "") for k in grid_keys]
    plt.bar(lrs, final_bwa, color="skyblue")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test BWA")
    plt.title("Final Test BWA per Learning-Rate (SPR_BENCH)")
    path = os.path.join(working_dir, "spr_bench_test_bwa_bar.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# ---------------------------- FIG 5 -----------------------------
try:
    if best_key:
        preds = np.array(exp[best_key]["predictions"])
        gts = np.array(exp[best_key]["ground_truth"])
        unique_labels = np.unique(np.concatenate([preds, gts]))[:20]  # at most 20
        pred_counts = [np.sum(preds == l) for l in unique_labels]
        gt_counts = [np.sum(gts == l) for l in unique_labels]
        x = np.arange(len(unique_labels))
        width = 0.35
        plt.figure(figsize=(8, 4))
        plt.bar(x - width / 2, gt_counts, width=width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width=width, label="Predictions")
        plt.xlabel("Label Index")
        plt.ylabel("Count")
        plt.title(f"Label Distribution – Best lr ({best_key.replace('lr_','')})")
        plt.legend()
        path = os.path.join(working_dir, "spr_bench_label_distribution.png")
        plt.savefig(path)
        print(f"Saved {path}")
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()
