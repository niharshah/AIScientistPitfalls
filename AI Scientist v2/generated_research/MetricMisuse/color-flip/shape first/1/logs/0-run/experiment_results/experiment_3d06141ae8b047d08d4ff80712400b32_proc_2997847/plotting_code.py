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


def _style(idx):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)], "-" if idx < len(colors) else "--"


for d_idx, (ds, ddict) in enumerate(experiment_data.items()):
    losses = ddict.get("losses", {})
    metrics_val = ddict.get("metrics", {}).get("val", [])
    swa = [t[0] for t in metrics_val] if metrics_val else []
    cwa = [t[1] for t in metrics_val] if metrics_val else []
    comp = [t[2] for t in metrics_val] if metrics_val else []

    # 1. pre-training loss
    try:
        if losses.get("pretrain"):
            plt.figure()
            plt.plot(
                range(1, len(losses["pretrain"]) + 1),
                losses["pretrain"],
                label="Pre-train",
                color=_style(d_idx)[0],
            )
            plt.title(f"{ds}: Pre-training Loss vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_pretrain_loss.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating pre-training loss plot for {ds}: {e}")
        plt.close()

    # 2. fine-tuning loss
    try:
        if losses.get("train") or losses.get("val"):
            plt.figure()
            if losses.get("train"):
                plt.plot(
                    range(1, len(losses["train"]) + 1),
                    losses["train"],
                    label="Train",
                    color=_style(d_idx)[0],
                    linestyle="-",
                )
            if losses.get("val"):
                plt.plot(
                    range(1, len(losses["val"]) + 1),
                    losses["val"],
                    label="Val",
                    color=_style(d_idx)[0],
                    linestyle="--",
                )
            plt.title(f"{ds}: Fine-tuning Loss (Train vs Val)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_finetune_loss.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating fine-tuning loss plot for {ds}: {e}")
        plt.close()

    # helper for metric curves
    def _plot_metric(values, metric_name, file_suffix):
        try:
            if values:
                plt.figure()
                plt.plot(
                    range(1, len(values) + 1),
                    values,
                    label=metric_name,
                    color=_style(d_idx)[0],
                )
                plt.title(f"{ds}: {metric_name} across Fine-tuning Epochs")
                plt.xlabel("Epoch")
                plt.ylabel(metric_name)
                plt.legend()
                fname = os.path.join(working_dir, f"{ds}_{file_suffix}.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating {metric_name} plot for {ds}: {e}")
            plt.close()

    # 3-5. metric curves
    _plot_metric(swa, "SWA", "SWA_curve")
    _plot_metric(cwa, "CWA", "CWA_curve")
    _plot_metric(comp, "CompWA", "CompWA_curve")
