import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data --------------------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp, run = {}, {}
else:
    run = exp.get("binary_no_counts", {}).get("spr_bench", {})

epochs = run.get("epochs", [])
train_ls = run.get("losses", {}).get("train", [])
dev_ls = run.get("losses", {}).get("dev", [])
train_pha = run.get("metrics", {}).get("train_PHA", [])
dev_pha = run.get("metrics", {}).get("dev_PHA", [])
test_m = run.get("test_metrics", {})
pred = run.get("predictions")
gt = run.get("ground_truth")

# -------------------- 1) loss curve --------------------
try:
    if epochs and train_ls and dev_ls:
        plt.figure()
        plt.plot(epochs, train_ls, label="Train")
        plt.plot(epochs, dev_ls, label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epoch (SPR_BENCH Binary Features)")
        plt.legend()
        save_path = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(save_path)
    else:
        raise ValueError("Loss data missing")
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    plt.close()

# -------------------- 2) PHA curve --------------------
try:
    if epochs and train_pha and dev_pha:
        plt.figure()
        plt.plot(epochs, train_pha, label="Train PHA")
        plt.plot(epochs, dev_pha, label="Dev PHA")
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.title("PHA vs. Epoch (SPR_BENCH Binary Features)")
        plt.legend()
        save_path = os.path.join(working_dir, "spr_bench_pha_curve.png")
        plt.savefig(save_path)
    else:
        raise ValueError("PHA data missing")
except Exception as e:
    print(f"Error creating PHA curve: {e}")
finally:
    plt.close()

# -------------------- 3) test metric bar plot --------------------
try:
    if test_m:
        metrics = list(test_m.keys())
        values = [test_m[m] for m in metrics]
        plt.figure()
        plt.bar(metrics, values, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Final Test Metrics (SPR_BENCH Binary Features)")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        save_path = os.path.join(working_dir, "spr_bench_test_metrics.png")
        plt.savefig(save_path)
    else:
        raise ValueError("Test metrics missing")
except Exception as e:
    print(f"Error creating test metric plot: {e}")
finally:
    plt.close()

# -------------------- print test metrics --------------------
if test_m:
    print("Test metrics:")
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")
