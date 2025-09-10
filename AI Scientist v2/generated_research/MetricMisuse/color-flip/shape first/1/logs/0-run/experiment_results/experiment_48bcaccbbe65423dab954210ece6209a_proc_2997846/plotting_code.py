import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helpers ----------
def _style(idx):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)], "-" if idx < len(colors) else "--"


# -------- iterate datasets ------------
for d_idx, (dname, dct) in enumerate(experiment_data.items()):
    losses = dct.get("losses", {})
    metrics_list = dct.get("metrics", {}).get("val", [])

    # 1. pre-training loss
    try:
        plt.figure()
        plt.plot(
            range(1, len(losses.get("pretrain", [])) + 1),
            losses.get("pretrain", []),
            color=_style(d_idx)[0],
            linestyle="-",
        )
        plt.title(f"{dname}: Pre-training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fname = os.path.join(working_dir, f"{dname}_pretrain_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error pretrain plot ({dname}): {e}")
        plt.close()

    # 2. fine-tune train/val losses
    try:
        plt.figure()
        tr, val = losses.get("train", []), losses.get("val", [])
        plt.plot(
            range(1, len(tr) + 1),
            tr,
            label="Train",
            color=_style(d_idx)[0],
            linestyle="-",
        )
        plt.plot(
            range(1, len(val) + 1),
            val,
            label="Val",
            color=_style(d_idx)[0],
            linestyle="--",
        )
        plt.title(f"{dname}: Fine-tuning Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_finetune_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error finetune plot ({dname}): {e}")
        plt.close()

    # helper to extract metric curves
    def _metric_curve(key):
        return [m.get(key, np.nan) for m in metrics_list]

    # 3-5. metric curves
    for m_key, suffix in [("SWA", "SWA"), ("CWA", "CWA"), ("CompWA", "CompWA")]:
        try:
            vals = _metric_curve(m_key)
            if not vals:  # skip if empty
                continue
            plt.figure()
            plt.plot(
                range(1, len(vals) + 1), vals, color=_style(d_idx)[0], linestyle="-"
            )
            plt.title(f"{dname}: {m_key} vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(m_key)
            fname = os.path.join(working_dir, f"{dname}_{suffix}_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error {m_key} plot ({dname}): {e}")
            plt.close()

    # print final-epoch metrics for quick inspection
    if metrics_list:
        print(f"{dname} final metrics:", metrics_list[-1])
