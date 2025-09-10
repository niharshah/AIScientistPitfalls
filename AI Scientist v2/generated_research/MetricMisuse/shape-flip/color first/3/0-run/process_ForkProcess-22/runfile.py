import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
ed = None
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["unidirectional_edges"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")


# ---- helper to fetch arrays ----
def get_metric_arr(metric_name):
    return [m[metric_name] for m in ed["metrics"]["train"]], [
        m[metric_name] for m in ed["metrics"]["val"]
    ]


if ed:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # 1) Loss plot
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) BWA plot
    try:
        tr, va = get_metric_arr("BWA")
        plt.figure()
        plt.plot(epochs, tr, label="Train")
        plt.plot(epochs, va, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title("SPR_BENCH: Training vs Validation BWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_bwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BWA plot: {e}")
        plt.close()

    # 3) CWA plot
    try:
        tr, va = get_metric_arr("CWA")
        plt.figure()
        plt.plot(epochs, tr, label="Train")
        plt.plot(epochs, va, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.title("SPR_BENCH: Training vs Validation CWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_cwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # 4) SWA plot
    try:
        tr, va = get_metric_arr("SWA")
        plt.figure()
        plt.plot(epochs, tr, label="Train")
        plt.plot(epochs, va, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH: Training vs Validation SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_swa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 5) StrWA plot
    try:
        tr, va = get_metric_arr("StrWA")
        plt.figure()
        plt.plot(epochs, tr, label="Train")
        plt.plot(epochs, va, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("StrWA")
        plt.title("SPR_BENCH: Training vs Validation StrWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_strwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating StrWA plot: {e}")
        plt.close()

    # ---- print final test metrics ----
    print("Final Test Metrics:", ed.get("test_metrics", {}))
