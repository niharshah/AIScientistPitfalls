import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs_dict = experiment_data.get("pretrain_epochs", {}).get("SPR_BENCH", {})
run_keys = sorted(runs_dict.keys(), key=lambda x: int(x))  # e.g. ['2','4','6',...]


# Helper to pick colors/linestyles that fit within default palette even for many lines
def _style(idx):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)], "-" if idx < len(colors) else "--"


# ---------- 1. Pre-training loss curves ----------
try:
    plt.figure()
    for i, k in enumerate(run_keys):
        losses = runs_dict[k]["losses"].get("pretrain", [])
        c, ls = _style(i)
        plt.plot(
            range(1, len(losses) + 1), losses, label=f"PT={k}", color=c, linestyle=ls
        )
    plt.title("SPR_BENCH: Pre-training Loss vs Epochs")
    plt.xlabel("Pre-training Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating pre-training loss plot: {e}")
    plt.close()

# ---------- 2. Fine-tuning loss curves ----------
try:
    plt.figure()
    for i, k in enumerate(run_keys):
        losses_train = runs_dict[k]["losses"].get("train", [])
        losses_val = runs_dict[k]["losses"].get("val", [])
        c, _ = _style(i)
        plt.plot(
            range(1, len(losses_train) + 1),
            losses_train,
            color=c,
            linestyle="-",
            label=f"Train (PT={k})",
        )
        plt.plot(
            range(1, len(losses_val) + 1),
            losses_val,
            color=c,
            linestyle="--",
            label=f"Val (PT={k})",
        )
    plt.title("SPR_BENCH: Fine-tuning Loss (Train vs Val)")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Loss")
    plt.legend(ncol=2, fontsize="small")
    fname = os.path.join(working_dir, "SPR_BENCH_finetune_loss.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating fine-tuning loss plot: {e}")
    plt.close()


# ---------- Metric plotting helper ----------
def plot_metric(metric_name, file_suffix):
    try:
        plt.figure()
        for i, k in enumerate(run_keys):
            vals = runs_dict[k]["metrics"].get(metric_name, [])
            c, ls = _style(i)
            plt.plot(
                range(1, len(vals) + 1),
                vals,
                label=f"{metric_name} (PT={k})",
                color=c,
                linestyle=ls,
            )
        plt.title(f"SPR_BENCH: {metric_name} across Fine-tuning Epochs")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel(metric_name)
        plt.legend(fontsize="small")
        fname = os.path.join(working_dir, f"SPR_BENCH_{file_suffix}.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating {metric_name} plot: {e}")
        plt.close()


# ---------- 3-5. Metric curves ----------
plot_metric("SWA", "SWA_curve")
plot_metric("CWA", "CWA_curve")
plot_metric("SCHM", "SCHM_curve")
