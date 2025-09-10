import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

hidden = experiment_data.get("hidden_dim_sweep", {})
keys = sorted(hidden.keys(), key=int)

# Helper to find best hidden dim (highest test HWA)
best_key, best_hwa = None, -1
for k in keys:
    hwa = hidden[k]["metrics"]["test"][2]
    if hwa > best_hwa:
        best_hwa, best_key = hwa, k

# 1) Train/Val loss curve for best hidden dim
try:
    d = hidden[best_key]
    fig, ax = plt.subplots()
    ax.plot(d["losses"]["train"], label="Train")
    ax.plot(d["losses"]["val"], label="Validation")
    ax.set_title(f"SPR_BENCH Loss Curve (hid={best_key})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    plt.savefig(os.path.join(working_dir, f"spr_loss_curve_hid{best_key}.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting loss curve: {e}")
    plt.close()

# 2) Train/Val HWA curve for best hidden dim
try:
    d = hidden[best_key]
    tr_hwa = [m[2] for m in d["metrics"]["train"]]
    val_hwa = [m[2] for m in d["metrics"]["val"]]
    fig, ax = plt.subplots()
    ax.plot(tr_hwa, label="Train HWA")
    ax.plot(val_hwa, label="Validation HWA")
    ax.set_title(f"SPR_BENCH HWA Curve (hid={best_key})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("HWA")
    ax.legend()
    plt.savefig(os.path.join(working_dir, f"spr_hwa_curve_hid{best_key}.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting HWA curve: {e}")
    plt.close()

# 3) Test HWA across hidden dims
try:
    hids = [int(k) for k in keys]
    hwa_vals = [hidden[k]["metrics"]["test"][2] for k in keys]
    fig, ax = plt.subplots()
    ax.bar(hids, hwa_vals, color="skyblue")
    ax.set_title("SPR_BENCH Test HWA vs Hidden Dim")
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Test HWA")
    plt.savefig(os.path.join(working_dir, "spr_test_hwa_by_hidden_dim.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting HWA vs hidden dim: {e}")
    plt.close()

# 4) Test SWA, CWA, HWA for best model
try:
    swa, cwa, hwa = hidden[best_key]["metrics"]["test"]
    fig, ax = plt.subplots()
    ax.bar(["SWA", "CWA", "HWA"], [swa, cwa, hwa], color=["orange", "green", "red"])
    ax.set_title(f"SPR_BENCH Test Metrics Breakdown (hid={best_key})")
    ax.set_ylabel("Score")
    plt.savefig(os.path.join(working_dir, f"spr_metric_breakdown_hid{best_key}.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting metric breakdown: {e}")
    plt.close()

# 5) Test loss across hidden dims
try:
    test_losses = [hidden[k]["losses"]["test"] for k in keys]
    fig, ax = plt.subplots()
    ax.bar(hids, test_losses, color="purple")
    ax.set_title("SPR_BENCH Test Loss vs Hidden Dim")
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Cross-Entropy Loss")
    plt.savefig(os.path.join(working_dir, "spr_test_loss_by_hidden_dim.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error plotting test loss: {e}")
    plt.close()
