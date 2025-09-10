import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- paths -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- load data -------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ft_dict = experiment_data.get("fine_tune_epochs", {})
variant_keys = sorted(ft_dict.keys())[
    :4
]  # safeguard, expect ['epochs_3', 'epochs_6', ...]
summary_schm = {}

# ----------- loss curves per variant -----------
for k in variant_keys:
    try:
        ed = ft_dict[k]
        plt.figure()
        plt.plot(ed["losses"]["train"], label="train")
        plt.plot(ed["losses"]["val"], label="val")
        plt.title(f"Loss Curve ({k})\nDataset: SPR_BENCH")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"loss_curve_{k}_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {k}: {e}")
        plt.close()

# ----------- metric curves per variant -----------
for k in variant_keys:
    try:
        ed = ft_dict[k]
        epochs = range(1, len(ed["metrics"]["SWA"]) + 1)
        plt.figure()
        plt.plot(epochs, ed["metrics"]["SWA"], label="SWA")
        plt.plot(epochs, ed["metrics"]["CWA"], label="CWA")
        plt.plot(epochs, ed["metrics"]["SCHM"], label="SCHM")
        plt.title(f"Metric Curves ({k})\nDataset: SPR_BENCH")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.legend()
        fname = f"metric_curves_{k}_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        # record final SCHM for summary
        summary_schm[k] = ed["metrics"]["SCHM"][-1] if ed["metrics"]["SCHM"] else 0.0
    except Exception as e:
        print(f"Error creating metric plot for {k}: {e}")
        plt.close()

# ----------- summary bar chart (final SCHM across variants) -----------
try:
    plt.figure()
    keys = list(summary_schm.keys())
    vals = [summary_schm[k] for k in keys]
    plt.bar(keys, vals, color="skyblue")
    plt.title("Final SCHM Across Fine-Tune Variants\nDataset: SPR_BENCH")
    plt.ylabel("SCHM")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fname = "final_schm_comparison_SPR_BENCH.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating summary SCHM plot: {e}")
    plt.close()

# ----------- print evaluation summary -----------
if summary_schm:
    best_variant = max(summary_schm, key=summary_schm.get)
    print("Final SCHM per variant:")
    for k, v in summary_schm.items():
        print(f"  {k}: {v:.3f}")
    print(f"Best variant: {best_variant} (SCHM={summary_schm[best_variant]:.3f})")
else:
    print("No summary metrics available.")
