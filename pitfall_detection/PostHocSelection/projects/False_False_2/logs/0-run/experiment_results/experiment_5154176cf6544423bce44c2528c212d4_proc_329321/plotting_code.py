import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

act_dict = experiment_data.get("activation_function", {})

# quick containers
epochs = None
loss_curves = {}
pha_curves = {}
test_metrics = {}

for act, bundle in act_dict.items():
    bench = bundle.get("spr_bench", {})
    losses = bench.get("losses", {})
    metrics = bench.get("metrics", {})
    test = bench.get("test_metrics", {})
    loss_curves[act] = (losses.get("train", []), losses.get("dev", []))
    pha_curves[act] = (metrics.get("train_PHA", []), metrics.get("dev_PHA", []))
    test_metrics[act] = test.get("PHA", np.nan)
    if epochs is None:
        epochs = list(range(1, len(losses.get("train", [])) + 1))

# ---------- figure 1: loss curves ----------
try:
    plt.figure()
    for act, (tr, dv) in loss_curves.items():
        plt.plot(epochs, tr, label=f"{act} train")
        plt.plot(epochs, dv, "--", label=f"{act} dev")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Synthetic: Training/Validation Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# ---------- figure 2: PHA curves ----------
try:
    plt.figure()
    for act, (tr, dv) in pha_curves.items():
        plt.plot(epochs, tr, label=f"{act} train")
        plt.plot(epochs, dv, "--", label=f"{act} dev")
    plt.xlabel("Epoch")
    plt.ylabel("PHA")
    plt.title("SPR_BENCH Synthetic: Training/Validation PHA Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_pha_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PHA curve figure: {e}")
    plt.close()

# ---------- figure 3: final test PHA ----------
try:
    plt.figure()
    acts = list(test_metrics.keys())
    scores = [test_metrics[a] for a in acts]
    plt.bar(acts, scores, color="skyblue")
    plt.ylabel("Test PHA")
    plt.title("SPR_BENCH Synthetic: Final Test PHA by Activation")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_test_pha_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test PHA bar chart: {e}")
    plt.close()

print("Final Test PHA:", test_metrics)
