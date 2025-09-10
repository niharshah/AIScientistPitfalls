import matplotlib.pyplot as plt
import numpy as np
import os

# --------- set up ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data -------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

embed_logs = experiment_data.get("embedding_dim", {})
dims = sorted(int(k.split("_")[-1]) for k in embed_logs)


def metric_list(dim_key, split, metric_name):
    return [
        ep_metrics[metric_name] for ep_metrics in embed_logs[dim_key]["metrics"][split]
    ]


# --------- choose best dimension by final val CpxWA ----------
best_dim = None
best_score = -1
for d in dims:
    key = f"dim_{d}"
    score = metric_list(key, "val", "cpx")[-1]
    if score > best_score:
        best_score, best_dim = score, d
best_key = f"dim_{best_dim}"

# --------- plot 1: Val CpxWA across dims ----------
try:
    plt.figure()
    for d in dims:
        key = f"dim_{d}"
        plt.plot(
            embed_logs[key]["epochs"],
            metric_list(key, "val", "cpx"),
            marker="o",
            label=f"dim{d}",
        )
    plt.title("Synthetic Dataset – Validation Complexity-Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    path = os.path.join(working_dir, "synth_val_cpxwa_vs_epoch_all_dims.png")
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"Error creating plot Val CpxWA: {e}")
    plt.close()

# --------- plot 2: Train vs Val CpxWA for best dim ----------
try:
    plt.figure()
    plt.plot(
        embed_logs[best_key]["epochs"],
        metric_list(best_key, "train", "cpx"),
        marker="o",
        label="Train",
    )
    plt.plot(
        embed_logs[best_key]["epochs"],
        metric_list(best_key, "val", "cpx"),
        marker="s",
        label="Validation",
    )
    plt.title(f"Synthetic Dataset – CpxWA (Best Embedding dim={best_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    path = os.path.join(working_dir, f"synth_cpxwa_train_val_best_dim{best_dim}.png")
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"Error creating plot Train/Val CpxWA: {e}")
    plt.close()

# --------- plot 3: Train Loss for best dim ----------
try:
    plt.figure()
    plt.plot(
        embed_logs[best_key]["epochs"],
        embed_logs[best_key]["losses"]["train"],
        marker="o",
    )
    plt.title(f"Synthetic Dataset – Training Loss (Best Embedding dim={best_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    path = os.path.join(working_dir, f"synth_train_loss_best_dim{best_dim}.png")
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"Error creating plot Train Loss: {e}")
    plt.close()

# --------- plot 4: Val ShapeWA across dims ----------
try:
    plt.figure()
    for d in dims:
        key = f"dim_{d}"
        plt.plot(
            embed_logs[key]["epochs"],
            metric_list(key, "val", "swa"),
            marker="o",
            label=f"dim{d}",
        )
    plt.title("Synthetic Dataset – Validation Shape-Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.legend()
    path = os.path.join(working_dir, "synth_val_swa_vs_epoch_all_dims.png")
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"Error creating plot Val ShapeWA: {e}")
    plt.close()

# --------- plot 5: Val ColorWA across dims ----------
try:
    plt.figure()
    for d in dims:
        key = f"dim_{d}"
        plt.plot(
            embed_logs[key]["epochs"],
            metric_list(key, "val", "cwa"),
            marker="o",
            label=f"dim{d}",
        )
    plt.title("Synthetic Dataset – Validation Color-Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.legend()
    path = os.path.join(working_dir, "synth_val_cwa_vs_epoch_all_dims.png")
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"Error creating plot Val ColorWA: {e}")
    plt.close()
