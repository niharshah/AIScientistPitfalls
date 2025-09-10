import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data.get("SPR_RGCN", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = {}


# helper
def save_close(fig_name):
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fig_name))
    plt.close()


# ---------- 1) loss curve ----------
try:
    plt.figure()
    ep = run["epochs"]
    plt.plot(ep, run["losses"]["train"], label="train")
    plt.plot(ep, run["losses"]["val"], "--", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_RGCN – Training vs Validation Loss")
    plt.legend()
    save_close("SPR_RGCN_loss_curve.png")
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- 2-4) metric curves ----------
for metric in ["CWA", "SWA", "CmpWA"]:
    try:
        plt.figure()
        ep = run["epochs"]
        plt.plot(ep, run["metrics"]["train"][metric], label="train")
        plt.plot(ep, run["metrics"]["val"][metric], "--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"SPR_RGCN – Training vs Validation {metric}")
        plt.legend()
        save_close(f"SPR_RGCN_{metric}_curve.png")
    except Exception as e:
        print(f"Error creating {metric} curve: {e}")
        plt.close()

# ---------- 5) test metrics summary ----------
try:
    plt.figure()
    names = ["Loss", "CWA", "SWA", "CmpWA"]
    values = [run["test_metrics"][k] for k in ["loss", "CWA", "SWA", "CmpWA"]]
    bars = plt.bar(names, values, color="skyblue")
    for b, v in zip(bars, values):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.3f}",
            ha="center",
            va="bottom",
        )
    plt.title("SPR_RGCN – Test Set Performance")
    save_close("SPR_RGCN_test_summary.png")
except Exception as e:
    print(f"Error creating test summary plot: {e}")
    plt.close()
