import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to pull metrics safely
def get_lists(d, keys):
    out = d
    for k in keys:
        out = out.get(k, [])
    return out


dataset_name = "SPR_BENCH"
batch_dict = experiment_data.get("BATCH_SIZE", {}).get(dataset_name, {})

# ----------------- 1-3: loss curves per batch size -----------------
for i, (batch_size, logs) in enumerate(batch_dict.items()):
    if i >= 3:  # safety, though we only have 3 batch sizes
        break
    try:
        train_loss = get_lists(logs, ["losses", "train"])
        val_loss = get_lists(logs, ["losses", "val"])
        epochs = list(range(1, len(train_loss) + 1))
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset_name} – Loss Curves (Batch {batch_size})")
        plt.legend()
        fname = f"{dataset_name}_batch{batch_size}_loss.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for batch {batch_size}: {e}")
        plt.close()

# ----------------- 4: validation SCWA curves combined --------------
try:
    plt.figure()
    for batch_size, logs in batch_dict.items():
        val_scwa = get_lists(logs, ["metrics", "val_SCWA"])
        if not val_scwa:
            continue
        epochs = list(range(1, len(val_scwa) + 1))
        plt.plot(epochs, val_scwa, marker="o", label=f"Batch {batch_size}")
    plt.xlabel("Epoch")
    plt.ylabel("SCWA")
    plt.title(f"{dataset_name} – Validation SCWA Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{dataset_name}_val_SCWA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating combined SCWA plot: {e}")
    plt.close()

# ----------------- 5: final SCWA vs batch size ---------------------
try:
    plt.figure()
    batch_sizes = []
    final_scores = []
    for batch_size, logs in batch_dict.items():
        scwa = get_lists(logs, ["metrics", "val_SCWA"])
        if scwa:
            batch_sizes.append(batch_size)
            final_scores.append(scwa[-1])
    if batch_sizes:
        plt.plot(batch_sizes, final_scores, marker="s", linestyle="-")
        plt.xlabel("Batch Size")
        plt.ylabel("Final SCWA (Epoch 5)")
        plt.title(f"{dataset_name} – Final SCWA vs. Batch Size")
        for x, y in zip(batch_sizes, final_scores):
            plt.text(x, y, f"{y:.3f}")
    plt.savefig(os.path.join(working_dir, f"{dataset_name}_final_SCWA_vs_batch.png"))
    plt.close()
except Exception as e:
    print(f"Error creating summary SCWA plot: {e}")
    plt.close()

print("Plotting complete; figures saved to", working_dir)
