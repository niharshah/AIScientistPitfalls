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


# helper to fetch arrays
def get_run(key):
    run = experiment_data["batch_size"][key]
    return (
        run["epochs"],
        run["losses"]["train"],
        run["losses"]["val"],
        run["metrics"]["train_macro_f1"],
        run["metrics"]["val_macro_f1"],
        run["predictions"],
        run["ground_truth"],
    )


runs = {}
for bs in [32, 64, 256]:
    k = f"SPR_BENCH_bs{bs}"
    if "batch_size" in experiment_data and k in experiment_data["batch_size"]:
        runs[bs] = get_run(k)

# 1) Loss curves
try:
    plt.figure(figsize=(6, 4))
    for bs, (ep, tr_loss, val_loss, *_) in runs.items():
        plt.plot(ep, tr_loss, label=f"train bs{bs}")
        plt.plot(ep, val_loss, "--", label=f"val bs{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Loss Curves across Batch Sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves_all_bs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# 2) Macro-F1 curves
try:
    plt.figure(figsize=(6, 4))
    for bs, (ep, *_, tr_f1, val_f1, _) in runs.items():
        plt.plot(ep, tr_f1, label=f"train bs{bs}")
        plt.plot(ep, val_f1, "--", label=f"val bs{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Macro-F1 Curves across Batch Sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves_all_bs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve figure: {e}")
    plt.close()

# 3) Final test Macro-F1 bar chart
try:
    plt.figure(figsize=(5, 4))
    bs_list, test_f1 = [], []
    from sklearn.metrics import f1_score

    for bs, (*_, preds, gts) in runs.items():
        bs_list.append(str(bs))
        test_f1.append(f1_score(gts, preds, average="macro"))
    plt.bar(bs_list, test_f1, color="skyblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Test Macro-F1")
    plt.title("SPR_BENCH: Final Test Macro-F1 by Batch Size")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test F1 bar figure: {e}")
    plt.close()
