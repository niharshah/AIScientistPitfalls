import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
exp_path_try = [
    os.path.join(working_dir, "experiment_data.npy"),
    "experiment_data.npy",
]
experiment_data = None
for p in exp_path_try:
    if os.path.isfile(p):
        experiment_data = np.load(p, allow_pickle=True).item()
        break
if experiment_data is None:
    raise FileNotFoundError("experiment_data.npy not found in expected locations.")

configs = ["multi_head", "single_head"]
colors = {"multi_head": "tab:blue", "single_head": "tab:orange"}


# ---------- helper: extract ----------
def get_losses(cfg, split):
    # returns epochs, losses
    l = experiment_data[cfg]["SPR_BENCH"]["losses"][split]
    return [t[1] for t in l], [t[2] for t in l]  # epoch, loss


def get_metric(cfg, idx):
    # idx: 0=cwa,1=swa,2=hwa,3=cna
    m = experiment_data[cfg]["SPR_BENCH"]["metrics"]["val"]
    return [t[1] for t in m], [t[2 + idx] for t in m]  # epoch, value


# ---------- 1) loss curves ----------
try:
    plt.figure(figsize=(10, 4))
    for cfg in configs:
        ep_tr, l_tr = get_losses(cfg, "train")
        ep_val, l_val = get_losses(cfg, "val")
        plt.plot(ep_tr, l_tr, "--", color=colors[cfg], label=f"{cfg} train")
        plt.plot(ep_val, l_val, "-", color=colors[cfg], label=f"{cfg} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Training / Validation Loss")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- 2) validation metrics ----------
metric_names = ["CWA", "SWA", "HWA", "CNA"]
try:
    plt.figure(figsize=(10, 6))
    for cfg in configs:
        for i, mname in enumerate(metric_names):
            ep, vals = get_metric(cfg, i)
            plt.plot(
                ep,
                vals,
                label=f"{cfg} {mname}",
                linestyle="-." if cfg == "single_head" else "-",
            )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH – Validation Metrics per Epoch")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_validation_metrics.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating validation metric plot: {e}")
    plt.close()

# ---------- 3) test metric bar chart ----------
try:
    width = 0.35
    x = np.arange(len(metric_names))  # positions
    plt.figure(figsize=(8, 4))
    for i, cfg in enumerate(configs):
        t = experiment_data[cfg]["SPR_BENCH"]["metrics"]["test"]
        vals = t[1:5]  # cwa,swa,hwa,cna
        plt.bar(x + i * width, vals, width, label=cfg, color=colors[cfg])
    plt.xticks(x + width / 2, metric_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("SPR_BENCH – Test Metrics Comparison")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_test_metric_bars.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()
