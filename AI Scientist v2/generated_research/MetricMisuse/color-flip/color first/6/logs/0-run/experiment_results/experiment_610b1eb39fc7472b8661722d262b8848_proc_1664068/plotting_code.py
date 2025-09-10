import matplotlib.pyplot as plt
import numpy as np
import os

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

bench_key = "SPR_BENCH"
runs = experiment_data.get("EPOCHS_tuning", {}).get(bench_key, {})

# ---------- gather metrics ----------
acc, cwa, swa, ewa = {}, {}, {}, {}
for rk, rdat in runs.items():
    gt = np.array(rdat.get("ground_truth", []))
    pr = np.array(rdat.get("predictions", []))
    acc[rk] = (gt == pr).mean() if gt.size else 0.0
    # last stored val metrics per run
    if rdat["metrics"]["val"]:
        _, metr = rdat["metrics"]["val"][-1]
        cwa[rk], swa[rk], ewa[rk] = metr["CWA"], metr["SWA"], metr["EWA"]

# ---------- 1) validation loss curves ----------
try:
    plt.figure()
    for rk, rdat in runs.items():
        x = [e for e, _ in rdat["losses"]["val"]]
        y = [l for _, l in rdat["losses"]["val"]]
        plt.plot(x, y, label=rk)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(
        "SPR_BENCH Validation Loss\n(Left: Ground Truth, Right: Generated Samples)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation-loss figure: {e}")
    plt.close()

# ---------- 2) training loss curves ----------
try:
    plt.figure()
    for rk, rdat in runs.items():
        x = [e for e, _ in rdat["losses"]["train"]]
        y = [l for _, l in rdat["losses"]["train"]]
        plt.plot(x, y, label=rk)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("SPR_BENCH Training Loss\n(Left: Ground Truth, Right: Generated Samples)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training-loss figure: {e}")
    plt.close()

# ---------- 3) weighted accuracy metrics ----------
try:
    plt.figure(figsize=(8, 4))
    x = np.arange(len(runs))
    bar_w = 0.2
    plt.bar(x - bar_w, [cwa.get(r, 0) for r in runs], width=bar_w, label="CWA")
    plt.bar(x, [swa.get(r, 0) for r in runs], width=bar_w, label="SWA")
    plt.bar(x + bar_w, [ewa.get(r, 0) for r in runs], width=bar_w, label="EWA")
    plt.xticks(x, runs.keys(), rotation=45)
    plt.ylabel("Score")
    plt.title(
        "SPR_BENCH Weighted Accuracies\n(Left: Ground Truth, Right: Generated Samples)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_weighted_accuracy_bars.png"))
    plt.close()
except Exception as e:
    print(f"Error creating weighted-accuracy figure: {e}")
    plt.close()

# ---------- 4) overall accuracy ----------
try:
    plt.figure()
    plt.bar(list(acc.keys()), list(acc.values()))
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(
        "SPR_BENCH Overall Accuracy\n(Left: Ground Truth, Right: Generated Samples)"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_bars.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy figure: {e}")
    plt.close()

# ---------- print summary ----------
print("\nFinal Metrics per run")
for rk in runs:
    print(
        f"{rk:10s} | ACC={acc.get(rk,0):.3f}  CWA={cwa.get(rk,0):.3f} "
        f"SWA={swa.get(rk,0):.3f}  EWA={ewa.get(rk,0):.3f}"
    )
