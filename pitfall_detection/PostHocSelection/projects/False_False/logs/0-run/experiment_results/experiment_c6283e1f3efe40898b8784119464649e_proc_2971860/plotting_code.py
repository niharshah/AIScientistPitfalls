import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# helper to load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
model_key, ds_key = "attention_only", "SPR_BENCH"
data = experiment_data.get(model_key, {}).get(ds_key, {})
losses = data.get("losses", {})
metrics = data.get("metrics", {})

# ------------------------------------------------------------------
# Plot 1: contrastive pretraining loss
try:
    plt.figure()
    plt.plot(
        range(1, len(losses.get("pretrain", [])) + 1),
        losses.get("pretrain", []),
        marker="o",
    )
    plt.title("SPR_BENCH: Contrastive Pretraining Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating pretraining loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 2: supervised train vs val loss
try:
    plt.figure()
    ep = range(1, len(losses.get("train", [])) + 1)
    plt.plot(ep, losses.get("train", []), label="Train")
    plt.plot(ep, losses.get("val", []), label="Validation")
    plt.title("SPR_BENCH: Supervised Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating train/val loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 3: validation metrics curves
try:
    plt.figure()
    ep = range(1, len(metrics.get("val_SWA", [])) + 1)
    plt.plot(ep, metrics.get("val_SWA", []), label="SWA")
    plt.plot(ep, metrics.get("val_CWA", []), label="CWA")
    plt.plot(ep, metrics.get("val_SCWA", []), label="SCWA")
    plt.title("SPR_BENCH: Validation Metrics over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric curves plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 4: best epoch metric summary
try:
    best_idx = int(np.argmax(metrics.get("val_SCWA", [0])))
    vals = [
        metrics.get("val_SWA", [0])[best_idx] if metrics.get("val_SWA") else 0,
        metrics.get("val_CWA", [0])[best_idx] if metrics.get("val_CWA") else 0,
        metrics.get("val_SCWA", [0])[best_idx] if metrics.get("val_SCWA") else 0,
    ]
    plt.figure()
    plt.bar(["SWA", "CWA", "SCWA"], vals, color=["skyblue", "salmon", "gold"])
    plt.title(f"SPR_BENCH: Best Epoch ({best_idx+1}) Metric Summary")
    plt.ylabel("Score")
    fname = os.path.join(working_dir, "SPR_BENCH_best_epoch_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best epoch bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print best metrics
if metrics.get("val_SCWA"):
    best = int(np.argmax(metrics["val_SCWA"]))
    print(f"Best epoch: {best+1}")
    print(f"SWA:  {metrics['val_SWA'][best]:.4f}")
    print(f"CWA:  {metrics['val_CWA'][best]:.4f}")
    print(f"SCWA: {metrics['val_SCWA'][best]:.4f}")
else:
    print("No metrics available to show.")
