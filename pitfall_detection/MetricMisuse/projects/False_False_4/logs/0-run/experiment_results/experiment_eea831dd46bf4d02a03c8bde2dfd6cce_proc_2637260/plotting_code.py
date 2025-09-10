import matplotlib.pyplot as plt
import numpy as np
import os

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

runs = experiment_data.get("SPR_BENCH", {}).get("runs", {})
max_plots = 5  # only plot first 5 runs to satisfy guideline
final_metrics = []

for i, (run_name, rec) in enumerate(runs.items()):
    if i >= max_plots:
        break
    epochs = list(range(1, len(rec["train_loss"]) + 1))

    # ---------- loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, rec["train_loss"], label="Train")
        plt.plot(epochs, rec["dev_loss"], label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("CrossEntropy")
        plt.title(f"SPR_BENCH ({run_name})\nLoss Curves")
        plt.legend()
        fname = f"spr_bench_{run_name}_loss.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {run_name}: {e}")
        plt.close()

    # ---------- accuracy curves ----------
    try:
        plt.figure()
        plt.plot(epochs, rec["train_acc"], label="Train")
        plt.plot(epochs, rec["dev_acc"], label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"SPR_BENCH ({run_name})\nAccuracy Curves")
        plt.legend()
        fname = f"spr_bench_{run_name}_accuracy.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting accuracy for {run_name}: {e}")
        plt.close()

    # ---------- WGMA curves ----------
    try:
        plt.figure()
        plt.plot(epochs, rec["wgma"], label="WGMA")
        plt.xlabel("Epoch")
        plt.ylabel("WGMA")
        plt.title(f"SPR_BENCH ({run_name})\nWGMA Progression")
        plt.legend()
        fname = f"spr_bench_{run_name}_wgma.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting WGMA for {run_name}: {e}")
        plt.close()

    # collect final metrics
    fm = rec.get("FINAL_TEST", {})
    final_metrics.append((run_name, fm.get("wgma", 0), fm.get("acc", 0)))

# ---------- aggregate bar chart ----------
try:
    if final_metrics:
        names, wgmas, accs = zip(*final_metrics)
        x = np.arange(len(names))
        plt.figure(figsize=(8, 4))
        plt.bar(x - 0.2, wgmas, width=0.4, label="Final WGMA")
        plt.bar(x + 0.2, accs, width=0.4, label="Final Accuracy")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel("Metric Value")
        plt.title("SPR_BENCH\nFinal Test Performance per Run")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_final_metrics.png"))
        plt.close()
except Exception as e:
    print(f"Error plotting aggregate metrics: {e}")
    plt.close()

# ---------- print numeric summary ----------
for run_name, wgma, acc in final_metrics:
    print(f"{run_name:25s} | Final WGMA: {wgma:.3f} | Final Acc: {acc:.3f}")
