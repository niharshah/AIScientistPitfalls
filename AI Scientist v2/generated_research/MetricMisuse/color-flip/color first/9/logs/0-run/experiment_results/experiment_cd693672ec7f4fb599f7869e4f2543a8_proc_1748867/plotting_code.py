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


# helper to unpack lists of tuples
def extract_curve(lst, idx_epoch=1, idx_val=2):
    epochs, vals = [], []
    for _, ep, v in lst:
        epochs.append(ep)
        vals.append(v)
    return epochs, vals


variants = ["SPR_BENCH_MEAN", "SPR_BENCH_CLS"]
colors = dict(SPR_BENCH_MEAN="tab:blue", SPR_BENCH_CLS="tab:orange")

# 1) loss curves ---------------------------------------------------------
try:
    plt.figure()
    for var in variants:
        ed = experiment_data["CLS_token_pooling"][var]
        ep_tr, tr = extract_curve(ed["losses"]["train"])
        ep_val, val = extract_curve(ed["losses"]["val"])
        plt.plot(ep_tr, tr, label=f"{var}-train", color=colors[var], linestyle="-")
        plt.plot(ep_val, val, label=f"{var}-val", color=colors[var], linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("CLS_token_pooling: Train vs. Val Loss")
    plt.legend()
    fname = os.path.join(working_dir, "CLS_token_pooling_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) validation HWA curves ----------------------------------------------
try:
    plt.figure()
    for var in variants:
        ed = experiment_data["CLS_token_pooling"][var]
        ep, hwa = [], []
        for _, ep_i, _, _, hwa_i, _ in ed["metrics"]["val"]:
            ep.append(ep_i)
            hwa.append(hwa_i)
        plt.plot(ep, hwa, label=var, color=colors[var])
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("CLS_token_pooling: Validation HWA")
    plt.legend()
    fname = os.path.join(working_dir, "CLS_token_pooling_val_HWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve: {e}")
    plt.close()

# 3) test metrics bar chart ---------------------------------------------
try:
    labels = ["HWA", "CNA"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    for i, var in enumerate(variants):
        _, hwa, _, _, cna = experiment_data["CLS_token_pooling"][var]["metrics"]["test"]
        vals = [hwa, cna]
        ax.bar(x + i * width, vals, width, label=var, color=colors[var])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("CLS_token_pooling: Test Metrics Comparison")
    ax.legend()
    fname = os.path.join(working_dir, "CLS_token_pooling_test_metrics_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar: {e}")
    plt.close()

# print test metrics -----------------------------------------------------
for var in variants:
    lr, hwa, swa, hwa2, cna = experiment_data["CLS_token_pooling"][var]["metrics"][
        "test"
    ]
    print(f"{var} TEST -> HWA={hwa:.3f} | CNA={cna:.3f}")
