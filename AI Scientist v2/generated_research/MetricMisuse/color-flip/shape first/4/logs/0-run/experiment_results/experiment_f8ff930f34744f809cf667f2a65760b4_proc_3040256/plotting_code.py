import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- helper for final metric table ----------
final_hwa = {}  # {(model, epochs): value}

models = ["LSTM", "BOE"]
dataset = "SPR_BENCH"
max_figs = 5
fig_count = 0

for model in models:
    mdl_dict = experiment_data.get(model, {}).get(dataset, {})
    # ----------------- LOSS CURVES FIG -----------------
    try:
        fig = plt.figure(figsize=(6, 4))
        for run_id, rec in mdl_dict.items():
            tr = rec["losses"]["train"]
            val = rec["losses"]["val"]
            xs = list(range(1, len(tr) + 1))
            plt.plot(xs, tr, linestyle="--", label=f"train_{run_id}ep")
            plt.plot(xs, val, linestyle="-", label=f"val_{run_id}ep")
            # store last hwa for table/bar
            final_hwa[(model, run_id)] = rec["metrics"]["val"][-1]
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{model} - {dataset} Training & Validation Loss")
        plt.legend(fontsize=7)
        out_path = os.path.join(working_dir, f"{model}_{dataset}_loss_curves.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        fig_count += 1
    except Exception as e:
        print(f"Error creating loss plot for {model}: {e}")
        plt.close()
    # ----------------- HWA CURVES FIG -----------------
    try:
        fig = plt.figure(figsize=(6, 4))
        for run_id, rec in mdl_dict.items():
            hwa = rec["metrics"]["val"]
            xs = list(range(1, len(hwa) + 1))
            plt.plot(xs, hwa, label=f"{run_id}ep")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title(f"{model} - {dataset} Validation HWA")
        plt.legend(fontsize=7)
        out_path = os.path.join(working_dir, f"{model}_{dataset}_hwa_curves.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        fig_count += 1
    except Exception as e:
        print(f"Error creating HWA plot for {model}: {e}")
        plt.close()

# ---------- BAR CHART COMPARISON (at most 5th fig) ----------
try:
    labels = []
    values = []
    for (m, ep), v in final_hwa.items():
        labels.append(f"{m}_{ep}")
        values.append(v)
    fig = plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="skyblue")
    plt.ylabel("Final-Epoch HWA")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Final HWA Comparison - {dataset}")
    plt.tight_layout()
    out_path = os.path.join(working_dir, f"{dataset}_final_hwa_comparison.png")
    plt.savefig(out_path)
    plt.close()
    fig_count += 1
except Exception as e:
    print(f"Error creating comparison bar chart: {e}")
    plt.close()

# ---------- PRINT METRIC TABLE ----------
print("\nFinal HWA scores:")
for (m, ep), v in sorted(final_hwa.items()):
    print(f"{m:4s} | {ep:>2s} epochs : {v:.3f}")

print(f"\nTotal figures created: {fig_count}")
