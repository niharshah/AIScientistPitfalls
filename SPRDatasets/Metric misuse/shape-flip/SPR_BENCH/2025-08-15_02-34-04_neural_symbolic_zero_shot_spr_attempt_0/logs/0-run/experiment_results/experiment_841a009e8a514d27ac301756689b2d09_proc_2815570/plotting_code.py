import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    lr_runs = experiment_data.get("learning_rate", {})
    best_key = None
    # detect best run (has "test" metrics/losses)
    for k, v in lr_runs.items():
        if "test" in v.get("metrics", {}):
            best_key = k
            break

    # 1) Loss curves for all learning rates
    try:
        plt.figure()
        for k, v in lr_runs.items():
            plt.plot(v["losses"]["train"], label=f"{k} train", alpha=0.6)
            plt.plot(v["losses"]["val"], label=f"{k} val", linestyle="--", alpha=0.6)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Loss per Epoch for each Learning Rate")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_all_lr.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss_all_lr plot: {e}")
        plt.close()

    # 2) Validation HWA curves for all learning rates
    try:
        plt.figure()
        for k, v in lr_runs.items():
            hwa_vals = [m[2] for m in v["metrics"]["val"]]
            plt.plot(hwa_vals, label=k)
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Validation HWA per Epoch (LR sweep)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_hwa_all_lr.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating hwa_all_lr plot: {e}")
        plt.close()

    # 3) Best run loss curves
    if best_key:
        try:
            plt.figure()
            best = lr_runs[best_key]
            plt.plot(best["losses"]["train"], label="train")
            plt.plot(best["losses"]["val"], label="validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"SPR_BENCH: Loss (Best {best_key})")
            plt.legend()
            fname = os.path.join(working_dir, f"spr_bench_loss_{best_key}.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating best_loss plot: {e}")
            plt.close()

    # 4) Best run metric curves
    if best_key:
        try:
            plt.figure()
            best = lr_runs[best_key]
            swa = [m[0] for m in best["metrics"]["val"]]
            cwa = [m[1] for m in best["metrics"]["val"]]
            hwa = [m[2] for m in best["metrics"]["val"]]
            plt.plot(swa, label="SWA")
            plt.plot(cwa, label="CWA")
            plt.plot(hwa, label="HWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"SPR_BENCH: Validation Metrics (Best {best_key})")
            plt.legend()
            fname = os.path.join(working_dir, f"spr_bench_metrics_{best_key}.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating best_metrics plot: {e}")
            plt.close()

    # 5) Bar chart of best run final test metrics
    if best_key and "test" in lr_runs[best_key]["metrics"]:
        try:
            plt.figure()
            test_met = lr_runs[best_key]["metrics"]["test"]
            names = ["SWA", "CWA", "HWA"]
            vals = list(test_met)
            plt.bar(names, vals, color=["skyblue", "lightgreen", "salmon"])
            plt.ylim(0, 1)
            plt.title(f"SPR_BENCH: Final Test Metrics (Best {best_key})")
            fname = os.path.join(working_dir, f"spr_bench_test_metrics_{best_key}.png")
            plt.savefig(fname)
            plt.close()
            print(f"Test metrics (best {best_key}):", dict(zip(names, vals)))
        except Exception as e:
            print(f"Error creating test_metrics bar plot: {e}")
            plt.close()
