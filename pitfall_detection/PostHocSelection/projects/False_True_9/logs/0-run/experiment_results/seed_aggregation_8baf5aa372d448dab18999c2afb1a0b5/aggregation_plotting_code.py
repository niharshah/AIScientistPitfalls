import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up output dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load all experiment_data.npy -----------
experiment_data_path_list = [
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_930038a9ae014668b759b437fdca3e7c_proc_3099458/experiment_data.npy",
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_f20704230f6d40cebfde3e31579e8f5e_proc_3099456/experiment_data.npy",
    "experiments/2025-08-16_02-30-16_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_ca25c2f21e3f4767a4586b917ec76994_proc_3099457/experiment_data.npy",
]

all_experiment_data = []
for exp_path in experiment_data_path_list:
    try:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), exp_path)
        exp_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading experiment data from {exp_path}: {e}")

if not all_experiment_data:
    print("No experiment data loaded; exiting.")
    exit()

# ---------- aggregate statistics across runs ----------
agg = {}  # {hs: {'tr': {ep: [v]}, 'val': {ep:[v]}, 'hwa': {ep:[v]}}}

for run in all_experiment_data:
    for hs, result in run.get("hidden_size", {}).items():
        rec = result.get("SPR_BENCH", {})
        tr_loss = rec.get("losses", {}).get("train", [])
        val_loss = rec.get("losses", {}).get("val", [])
        hwa_vals = [(e, h) for e, _, _, h in rec.get("metrics", {}).get("val", [])]

        if not (tr_loss and val_loss and hwa_vals):
            continue  # skip incomplete runs

        hs_dict = agg.setdefault(hs, {"tr": {}, "val": {}, "hwa": {}})
        for ep, v in tr_loss:
            hs_dict["tr"].setdefault(ep, []).append(v)
        for ep, v in val_loss:
            hs_dict["val"].setdefault(ep, []).append(v)
        for ep, h in hwa_vals:
            hs_dict["hwa"].setdefault(ep, []).append(h)

# sort hidden sizes for consistent plotting
sorted_hs = sorted(agg.keys())


# helper to compute mean and se arrays
def mean_se(dic):
    eps = sorted(dic.keys())
    vals = [dic[ep] for ep in eps]
    mean = np.array([np.mean(v) for v in vals])
    se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) for v in vals])
    return np.array(eps), mean, se


# ---------- plot 1: loss curves mean ± SE -------------
try:
    plt.figure()
    for hs in sorted_hs:
        ep_tr, mu_tr, se_tr = mean_se(agg[hs]["tr"])
        ep_val, mu_val, se_val = mean_se(agg[hs]["val"])
        plt.plot(ep_tr, mu_tr, label=f"train hs={hs}")
        plt.fill_between(ep_tr, mu_tr - se_tr, mu_tr + se_tr, alpha=0.2)
        plt.plot(ep_val, mu_val, linestyle="--", label=f"val hs={hs}")
        plt.fill_between(ep_val, mu_val - se_val, mu_val + se_val, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Mean ± SE Loss (Hidden-Size Sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_mean_se.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# ---------- plot 2: HWA curves mean ± SE -------------
try:
    plt.figure()
    for hs in sorted_hs:
        ep_hwa, mu_hwa, se_hwa = mean_se(agg[hs]["hwa"])
        plt.plot(ep_hwa, mu_hwa, label=f"hs={hs}")
        plt.fill_between(ep_hwa, mu_hwa - se_hwa, mu_hwa + se_hwa, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Mean ± SE HWA Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves_mean_se.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated HWA plot: {e}")
    plt.close()

# ---------- plot 3: final HWA bar w/ error -------------
try:
    plt.figure()
    hs_labels, means, ses = [], [], []
    for hs in sorted_hs:
        # final epoch assumed to be max key
        final_ep = max(agg[hs]["hwa"].keys())
        vals = agg[hs]["hwa"][final_ep]
        hs_labels.append(str(hs))
        means.append(np.mean(vals))
        ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    x = np.arange(len(hs_labels))
    plt.bar(x, means, yerr=ses, capsize=5, color="skyblue")
    plt.xticks(x, hs_labels)
    plt.xlabel("Hidden Size")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_BENCH: Final HWA by Hidden Size (Mean ± SE)")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar_mean_se.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar chart: {e}")
    plt.close()

# ---------- textual summary ----------------
print("Final-epoch HWA (mean ± SE) per hidden size:")
for hs, label in zip(sorted_hs, hs_labels):
    m, s = means[hs_labels.index(label)], ses[hs_labels.index(label)]
    print(f"  hidden={hs:>3}: {m:.4f} ± {s:.4f}")
