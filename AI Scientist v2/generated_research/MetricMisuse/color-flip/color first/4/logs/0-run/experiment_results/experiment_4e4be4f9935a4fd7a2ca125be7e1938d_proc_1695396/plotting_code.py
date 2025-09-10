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

exp_names = list(experiment_data.keys())[:2]  # safeguard, only need first two
fig_count = 0

# ----- per-experiment plots -----
for exp in exp_names:
    spr_data = experiment_data[exp]["SPR_BENCH"]
    epochs = np.arange(1, len(spr_data["losses"]["train"]) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, spr_data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, spr_data["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{exp} - SPR_BENCH Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{exp}_SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
        fig_count += 1
    except Exception as e:
        print(f"Error creating loss plot for {exp}: {e}")
        plt.close()

    # 2) Metric curves
    try:
        plt.figure()
        vals = spr_data["metrics"]["val"]
        acc = [d["acc"] for d in vals]
        cwa = [d["CWA"] for d in vals]
        swa = [d["SWA"] for d in vals]
        comp = [d["CompWA"] for d in vals]
        plt.plot(epochs, acc, label="ACC")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, comp, label="CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{exp} - SPR_BENCH Validation Metrics")
        plt.legend()
        fname = os.path.join(working_dir, f"{exp}_SPR_BENCH_metric_curves.png")
        plt.savefig(fname)
        plt.close()
        fig_count += 1
    except Exception as e:
        print(f"Error creating metric plot for {exp}: {e}")
        plt.close()

# ----- comparison plot -----
try:
    plt.figure()
    finals = [
        experiment_data[e]["SPR_BENCH"]["metrics"]["val"][-1]["acc"] for e in exp_names
    ]
    plt.bar(exp_names, finals)
    plt.ylabel("Final Accuracy")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH Final Accuracy Comparison")
    fname = os.path.join(working_dir, "SPR_BENCH_final_accuracy_comparison.png")
    plt.savefig(fname)
    plt.close()
    fig_count += 1
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()

print(f"Done. Generated {fig_count} figure(s).")
