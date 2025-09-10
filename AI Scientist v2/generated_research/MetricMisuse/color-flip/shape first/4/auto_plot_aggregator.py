"""
Final Aggregated Figures Script for Context-aware Contrastive Learning Paper
This script loads experiment .npy files from multiple experiments (baseline, research, ablation)
and produces final publication–ready figures, saved to the 'figures/' folder.
Each figure is produced in its own try–except block so that failure in one plot does not affect others.
All plots are made with clear labels, legends, titles and larger font sizes.
Only data stored in the provided .npy files are used.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# For publication-quality fonts
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})

# Create output directory for figures
os.makedirs("figures", exist_ok=True)

def load_experiment_data(filepath):
    try:
        data = np.load(filepath, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# Helper: get representative final HWA value from a run dictionary for a given key and epoch label.
def get_final_hwa(run_dict, epoch_key):
    try:
        # run_dict should be a dict keyed by epochs (as strings) or directly available list.
        if epoch_key in run_dict:
            rec = run_dict[epoch_key]
        else:
            # if not found, choose the maximum epoch key available
            keys = list(run_dict.keys())
            if not keys:
                return None
            rec = run_dict[max(keys, key=lambda x: int(x))]
        return rec["metrics"]["val"][-1]
    except Exception as e:
        print(f"Error extracting final HWA for epoch {epoch_key}: {e}")
        return None

# -------------------------------------------------------------
# Figure 1: Baseline Loss Curves (from fileA)
# File A (baseline and research summary; same file):
fileA = "experiment_results/experiment_05c15e4cd885474784d82b668a6f6b01_proc_3023913/experiment_data.npy"
dataA = load_experiment_data(fileA)
try:
    runs = dataA.get("epochs", {}).get("SPR_BENCH", {})
    plt.figure(figsize=(8,6))
    for epoch_str, rec in runs.items():
        epochs = range(1, len(rec["losses"]["train"]) + 1)
        plt.plot(epochs, rec["losses"]["train"], linestyle="--", label=f"{epoch_str} Epoch Train")
        plt.plot(epochs, rec["losses"]["val"], label=f"{epoch_str} Epoch Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR BENCH: Baseline Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join("figures", "Baseline_Loss_Curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 1 (Baseline Loss Curves): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 2: Baseline HWA Curves (from fileA)
try:
    plt.figure(figsize=(8,6))
    for epoch_str, rec in runs.items():
        epochs = range(1, len(rec["metrics"]["val"]) + 1)
        plt.plot(epochs, rec["metrics"]["val"], label=f"{epoch_str} Epoch HWA")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR BENCH: Baseline HWA Across Epochs")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join("figures", "Baseline_HWA_Curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 2 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 2 (Baseline HWA Curves): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 3: Bag-of-Embeddings Loss Curves (fileB)
fileB = "experiment_results/experiment_f8ff930f34744f809cf667f2a65760b4_proc_3040256/experiment_data.npy"
dataB = load_experiment_data(fileB)
try:
    # dataB contains keys "LSTM" and "BOE" for the SPR_BENCH dataset.
    models = ["LSTM", "BOE"]
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    for i, model in enumerate(models):
        mdl_dict = dataB.get(model, {}).get("SPR_BENCH", {})
        for run_id, rec in mdl_dict.items():
            xs = range(1, len(rec["losses"]["train"]) + 1)
            axes[i].plot(xs, rec["losses"]["train"], linestyle="--", label=f"Train {run_id} Epoch")
            axes[i].plot(xs, rec["losses"]["val"], label=f"Val {run_id} Epoch")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Cross-Entropy Loss")
        axes[i].set_title(f"{model} Loss Curves")
        axes[i].legend()
    plt.suptitle("Bag-of-Embeddings Classifier: Loss Curves (SPR BENCH)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join("figures", "BOE_Loss_Curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 3 (BOE Loss Curves): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 4: Bag-of-Embeddings HWA and Final Comparison
try:
    models = ["LSTM", "BOE"]
    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    final_hwa = {}
    # Left: LSTM HWA curves
    lstm_dict = dataB.get("LSTM", {}).get("SPR_BENCH", {})
    for run_id, rec in lstm_dict.items():
        xs = range(1, len(rec["metrics"]["val"]) + 1)
        axes[0].plot(xs, rec["metrics"]["val"], label=f"{run_id} Epoch")
        final_hwa[f"LSTM_{run_id}"] = rec["metrics"]["val"][-1]
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("HWA")
    axes[0].set_title("LSTM HWA Curves")
    axes[0].legend()
    # Middle: BOE HWA curves
    boe_dict = dataB.get("BOE", {}).get("SPR_BENCH", {})
    for run_id, rec in boe_dict.items():
        xs = range(1, len(rec["metrics"]["val"]) + 1)
        axes[1].plot(xs, rec["metrics"]["val"], label=f"{run_id} Epoch")
        final_hwa[f"BOE_{run_id}"] = rec["metrics"]["val"][-1]
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HWA")
    axes[1].set_title("BOE HWA Curves")
    axes[1].legend()
    # Right: Final HWA Comparison Bar Chart
    labels = list(final_hwa.keys())
    values = [final_hwa[k] for k in labels]
    axes[2].bar(labels, values, color="skyblue")
    axes[2].set_xlabel("Configuration")
    axes[2].set_ylabel("Final HWA")
    axes[2].set_title("Final HWA Comparison")
    axes[2].tick_params(axis="x", rotation=45)
    plt.suptitle("Bag-of-Embeddings: HWA Metrics (SPR BENCH)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join("figures", "BOE_HWA_Comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 4 (BOE HWA Comparison): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 5: Factorized Shape-and-Color Embeddings 
fileC = "experiment_results/experiment_eb75560787494effa0d1b600ca95c534_proc_3040258/experiment_data.npy"
dataC = load_experiment_data(fileC)
try:
    # dataC has keys "baseline" and "factorized"; use epoch "10" as representative.
    epoch_key = "10"
    rec_base = dataC.get("baseline", {}).get("SPR_BENCH", {}).get(epoch_key, None)
    rec_fact = dataC.get("factorized", {}).get("SPR_BENCH", {}).get(epoch_key, None)
    if rec_base is None or rec_fact is None:
        raise ValueError("Missing epoch '10' run for baseline or factorized.")
    epochs = range(1, len(rec_base["losses"]["train"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    # Left: Loss curves comparison
    axes[0].plot(epochs, rec_base["losses"]["train"], linestyle="--", label="Baseline Train")
    axes[0].plot(epochs, rec_base["losses"]["val"], label="Baseline Val")
    axes[0].plot(epochs, rec_fact["losses"]["train"], linestyle="--", label="Factorized Train")
    axes[0].plot(epochs, rec_fact["losses"]["val"], label="Factorized Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    # Right: HWA curves comparison
    axes[1].plot(epochs, rec_base["metrics"]["val"], marker="o", label="Baseline HWA")
    axes[1].plot(epochs, rec_fact["metrics"]["val"], marker="s", label="Factorized HWA")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HWA")
    axes[1].set_title("HWA Curves")
    axes[1].legend()
    plt.suptitle("Factorized vs Baseline (SPR BENCH) [10 Epoch Run]")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join("figures", "Factorized_Comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 5 (Factorized Comparison): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 6: Padding-Mask Removal Ablation
fileD = "experiment_results/experiment_6279441767534ae892b597c077c2572d_proc_3040259/experiment_data.npy"
dataD = load_experiment_data(fileD)
try:
    # dataD has keys "baseline" and "padding_mask_removal" for SPR_BENCH.
    runs_baseline = dataD.get("baseline", {}).get("SPR_BENCH", {})
    runs_nomask = dataD.get("padding_mask_removal", {}).get("SPR_BENCH", {})
    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    # Subplot 1: Baseline loss curves
    for ep, rec in sorted(runs_baseline.items(), key=lambda x: int(x[0])):
        xs = range(1, len(rec["losses"]["train"]) + 1)
        axes[0].plot(xs, rec["losses"]["train"], label=f"{ep} Epoch Train")
        axes[0].plot(xs, rec["losses"]["val"], linestyle="--", label=f"{ep} Epoch Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Baseline (Packed) Loss")
    axes[0].legend(fontsize=10)
    # Subplot 2: No-Mask loss curves
    for ep, rec in sorted(runs_nomask.items(), key=lambda x: int(x[0])):
        xs = range(1, len(rec["losses"]["train"]) + 1)
        axes[1].plot(xs, rec["losses"]["train"], label=f"{ep} Epoch Train")
        axes[1].plot(xs, rec["losses"]["val"], linestyle="--", label=f"{ep} Epoch Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("No Padding-Mask Loss")
    axes[1].legend(fontsize=10)
    # Subplot 3: Final HWA comparison bar chart
    final_hwa_base = []
    final_hwa_nomask = []
    epoch_vals = []
    common_epochs = set(runs_baseline.keys()) & set(runs_nomask.keys())
    for ep in sorted(common_epochs, key=lambda x: int(x)):
        epoch_vals.append(ep)
        final_hwa_base.append(runs_baseline[ep]["metrics"]["val"][-1])
        final_hwa_nomask.append(runs_nomask[ep]["metrics"]["val"][-1])
    width = 0.35
    idx = range(len(epoch_vals))
    axes[2].bar([i - width/2 for i in idx], final_hwa_base, width, label="Baseline")
    axes[2].bar([i + width/2 for i in idx], final_hwa_nomask, width, label="NoMask")
    axes[2].set_xticks(idx)
    axes[2].set_xticklabels(epoch_vals)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Final HWA")
    axes[2].set_title("Final HWA Comparison")
    axes[2].legend(fontsize=10)
    plt.suptitle("Padding-Mask Removal Ablation (SPR BENCH)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_path = os.path.join("figures", "PaddingMask_Removal.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 6 (Padding-Mask Removal): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 7: Frozen-Embedding Ablation (select epoch 10)
fileE = "experiment_results/experiment_996ff291e5c14f6dab03fc25b1490baf_proc_3040256/experiment_data.npy"
dataE = load_experiment_data(fileE)
try:
    # dataE has keys "baseline" and "frozen_emb" for SPR_BENCH; choose epoch "10"
    epoch_key = "10"
    rec_base = dataE.get("baseline", {}).get("SPR_BENCH", {}).get(epoch_key, None)
    rec_frozen = dataE.get("frozen_emb", {}).get("SPR_BENCH", {}).get(epoch_key, None)
    if rec_base is None or rec_frozen is None:
        raise ValueError("Missing epoch '10' run for baseline or frozen_emb.")
    epochs = range(1, len(rec_base["losses"]["train"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    # Left: Loss curves
    axes[0].plot(epochs, rec_base["losses"]["train"], linestyle="--", label="Baseline Train")
    axes[0].plot(epochs, rec_base["losses"]["val"], label="Baseline Val")
    axes[0].plot(epochs, rec_frozen["losses"]["train"], linestyle="--", label="Frozen Train")
    axes[0].plot(epochs, rec_frozen["losses"]["val"], label="Frozen Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves (Epoch 10)")
    axes[0].legend()
    # Right: HWA curves
    axes[1].plot(epochs, rec_base["metrics"]["val"], marker="o", label="Baseline HWA")
    axes[1].plot(epochs, rec_frozen["metrics"]["val"], marker="s", label="Frozen HWA")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HWA")
    axes[1].set_title("HWA Curves (Epoch 10)")
    axes[1].legend()
    plt.suptitle("Frozen-Embedding Ablation (SPR BENCH, 10 Epochs)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join("figures", "FrozenEmbedding_Ablation.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 7 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 7 (Frozen-Embedding Ablation): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 8: Shape-Color Split Tokenization Ablation (select epoch 10)
fileF = "experiment_results/experiment_94d8a87f991c48c6834cddea84dfde40_proc_3040257/experiment_data.npy"
dataF = load_experiment_data(fileF)
try:
    runs = dataF.get("shape_color_split", {}).get("SPR_BENCH", {})
    # Choose epoch "10" if available
    epoch_key = "10"
    rec = runs.get(epoch_key, None)
    if rec is None:
        raise ValueError("Missing epoch '10' run for shape_color_split.")
    epochs = range(1, len(rec["losses"]["train"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8,10))
    axes[0].plot(epochs, rec["losses"]["train"], linestyle="--", label="Train Loss")
    axes[0].plot(epochs, rec["losses"]["val"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[1].plot(epochs, rec["metrics"]["val"], marker="o", color="green", label="Val HWA")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HWA")
    axes[1].set_title("HWA Curve")
    axes[1].legend()
    plt.suptitle("Shape-Color Split Tokenization Ablation (SPR BENCH, 10 Epochs)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join("figures", "ShapeColor_Split.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 8 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 8 (Shape-Color Split Tokenization): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 9: Token-Order Randomization Ablation
fileG = "experiment_results/experiment_7bb4990a50d54779b09f9b42c40e3e2a_proc_3040259/experiment_data.npy"
dataG = load_experiment_data(fileG)
try:
    # dataG contains keys "baseline" and "token_order_randomization" for SPR_BENCH.
    runs_base = dataG.get("baseline", {}).get("SPR_BENCH", {})
    runs_rand = dataG.get("token_order_randomization", {}).get("SPR_BENCH", {})
    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    # Subplot 1: Loss curves (baseline vs randomized)
    for ep, rec in sorted(runs_base.items(), key=lambda x: int(x[0])):
        xs = range(1, len(rec["losses"]["train"]) + 1)
        axes[0].plot(xs, rec["losses"]["train"], label=f"B {ep}ep Train")
        axes[0].plot(xs, rec["losses"]["val"], linestyle="--", label=f"B {ep}ep Val")
    for ep, rec in sorted(runs_rand.items(), key=lambda x: int(x[0])):
        xs = range(1, len(rec["losses"]["train"]) + 1)
        axes[0].plot(xs, rec["losses"]["train"], linestyle=":", label=f"R {ep}ep Train")
        axes[0].plot(xs, rec["losses"]["val"], linestyle="-.", label=f"R {ep}ep Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves Comparison")
    axes[0].legend(fontsize=8)
    # Subplot 2: HWA curves comparison
    for ep, rec in sorted(runs_base.items(), key=lambda x: int(x[0])):
        xs = range(1, len(rec["metrics"]["val"]) + 1)
        axes[1].plot(xs, rec["metrics"]["val"], label=f"B {ep}ep")
    for ep, rec in sorted(runs_rand.items(), key=lambda x: int(x[0])):
        xs = range(1, len(rec["metrics"]["val"]) + 1)
        axes[1].plot(xs, rec["metrics"]["val"], label=f"R {ep}ep")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("HWA")
    axes[1].set_title("HWA Curves Comparison")
    axes[1].legend(fontsize=8)
    # Subplot 3: Final HWA vs Epochs
    common_epochs = sorted(set(int(ep) for ep in runs_base) & set(int(ep) for ep in runs_rand))
    final_base = [runs_base[str(ep)]["metrics"]["val"][-1] for ep in common_epochs]
    final_rand = [runs_rand[str(ep)]["metrics"]["val"][-1] for ep in common_epochs]
    axes[2].plot(common_epochs, final_base, marker="o", label="Baseline Final HWA")
    axes[2].plot(common_epochs, final_rand, marker="s", label="Randomized Final HWA")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Final HWA")
    axes[2].set_title("Final HWA Comparison")
    axes[2].legend(fontsize=8)
    plt.suptitle("Token-Order Randomization Ablation (SPR BENCH)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join("figures", "TokenOrder_Randomization.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 9 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 9 (Token-Order Randomization): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 10: Bidirectional LSTM Ablation
fileH = "experiment_results/experiment_1749558b632141b096b5eabcf0cac7fc_proc_3040258/experiment_data.npy"
dataH = load_experiment_data(fileH)
try:
    models_bi = ["UNI_LSTM", "BI_LSTM"]
    dataset = "SPR_BENCH"
    fig, axes = plt.subplots(2, 2, figsize=(16,12))
    # Subplot 1: Final HWA vs Epochs for both models
    for m in models_bi:
        keys = dataH.get(m, {}).get(dataset, {}).keys()
        if not keys:
            continue
        epochs = sorted([int(k) for k in keys])
        hwa = [dataH[m][dataset][str(ep)]["metrics"]["val"][-1] for ep in epochs]
        axes[0,0].plot(epochs, hwa, marker="o", label=m)
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Final HWA")
    axes[0,0].set_title("Final HWA vs Epochs")
    axes[0,0].legend()
    # Helper function: get longest-run record for a model
    def longest_run(model):
        model_data = dataH.get(model, {}).get(dataset, {})
        if not model_data:
            return None, None
        ep = max(int(k) for k in model_data.keys())
        return ep, model_data.get(str(ep), None)
    # Subplot 2: UNI_LSTM Loss Curves (longest run)
    ep_uni, rec_uni = longest_run("UNI_LSTM")
    if rec_uni is not None:
        epochs_uni = range(1, len(rec_uni["losses"]["train"]) + 1)
        axes[0,1].plot(epochs_uni, rec_uni["losses"]["train"], linestyle="--", label="Train")
        axes[0,1].plot(epochs_uni, rec_uni["losses"]["val"], label="Validation")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("Loss")
        axes[0,1].set_title(f"UNI_LSTM Loss Curves ({ep_uni} Epochs)")
        axes[0,1].legend()
    # Subplot 3: BI_LSTM Loss Curves (longest run)
    ep_bi, rec_bi = longest_run("BI_LSTM")
    if rec_bi is not None:
        epochs_bi = range(1, len(rec_bi["losses"]["train"]) + 1)
        axes[1,0].plot(epochs_bi, rec_bi["losses"]["train"], linestyle="--", label="Train")
        axes[1,0].plot(epochs_bi, rec_bi["losses"]["val"], label="Validation")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("Loss")
        axes[1,0].set_title(f"BI_LSTM Loss Curves ({ep_bi} Epochs)")
        axes[1,0].legend()
    # Subplot 4: Per-epoch HWA curves for longest runs of both models
    if rec_uni is not None and rec_bi is not None:
        axes[1,1].plot(epochs_uni, rec_uni["metrics"]["val"], marker="o", label=f"UNI_LSTM ({ep_uni}e)")
        axes[1,1].plot(epochs_bi, rec_bi["metrics"]["val"], marker="s", label=f"BI_LSTM ({ep_bi}e)")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("HWA")
        axes[1,1].set_title("Per-Epoch HWA Comparison")
        axes[1,1].legend()
    plt.suptitle("Bidirectional LSTM Ablation (SPR BENCH)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join("figures", "Bidirectional_LSTM.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 10 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 10 (Bidirectional LSTM Ablation): {e}")
    plt.close()

# -------------------------------------------------------------
# Figure 11: Final HWA Comparison Across Experiments
# For each experiment file, extract a representative final HWA
def extract_experiment_final_hwa(data, key_path, rep_epoch="10"):
    """
    key_path: list of keys to traverse. For some experiments the value is under:
      - For fileA: ["epochs", "SPR_BENCH"]
      - For fileB: for Bag-of-Embeddings use key, e.g., "LSTM"
      - For fileC: use "baseline" etc.
    We return the final HWA value for the rep_epoch if available, otherwise the highest epoch.
    """
    try:
        # Traverse the dict according to key_path
        d = data
        for key in key_path:
            d = d.get(key, {})
        if rep_epoch in d:
            rec = d[rep_epoch]
        else:
            # choose maximum key based on integer
            if not d:
                return None
            rec = d[max(d.keys(), key=lambda x: int(x))]
        return rec["metrics"]["val"][-1]
    except Exception as e:
        return None

final_scores = {}
# Baseline from fileA, key_path = ["epochs", "SPR_BENCH"]
if dataA:
    score = extract_experiment_final_hwa(dataA, ["epochs", "SPR_BENCH"], "10")
    if score is not None:
        final_scores["Baseline"] = score
# Bag-of-Embeddings (LSTM) from fileB, key_path = ["LSTM", "SPR_BENCH"]
if dataB:
    score = extract_experiment_final_hwa(dataB, ["LSTM", "SPR_BENCH"], "10")
    if score is not None:
        final_scores["BOE-LSTM"] = score
# Factorized: use baseline key from fileC
if dataC:
    score = extract_experiment_final_hwa(dataC, ["baseline", "SPR_BENCH"], "10")
    if score is not None:
        final_scores["Factorized (Base)"] = score
# Padding-Mask Removal: use baseline from fileD
if dataD:
    score = extract_experiment_final_hwa(dataD, ["baseline", "SPR_BENCH"], "10")
    if score is not None:
        final_scores["PadMask Baseline"] = score
# Frozen-Embedding: from fileE baseline.
if dataE:
    score = extract_experiment_final_hwa(dataE, ["baseline", "SPR_BENCH"], "10")
    if score is not None:
        final_scores["FrozenEmb Baseline"] = score
# Shape-Color Split: from fileF
if dataF:
    score = extract_experiment_final_hwa(dataF, ["shape_color_split", "SPR_BENCH"], "10")
    if score is not None:
        final_scores["Shape-Color Split"] = score
# Token-Order Randomization: from fileG (baseline)
if dataG:
    score = extract_experiment_final_hwa(dataG, ["baseline", "SPR_BENCH"], "10")
    if score is not None:
        final_scores["Token-Order Base"] = score
# Bidirectional: from fileH, use UNI_LSTM, choose maximum epoch.
if dataH:
    try:
        mdl_data = dataH.get("UNI_LSTM", {}).get("SPR_BENCH", {})
        if mdl_data:
            rep = mdl_data[max(mdl_data.keys(), key=lambda x: int(x))]
            final_scores["Uni LSTM"] = rep["metrics"]["val"][-1]
    except Exception:
        pass

try:
    labels = list(final_scores.keys())
    values = [final_scores[l] for l in labels]
    plt.figure(figsize=(10,6))
    plt.bar(labels, values, color="mediumpurple")
    plt.xlabel("Experiment")
    plt.ylabel("Final HWA")
    plt.title("Final HWA Comparison Across Experiments")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join("figures", "Final_HWA_Comparison_All.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 11 to {out_path}")
except Exception as e:
    print(f"Error creating Figure 11 (Final HWA Comparison): {e}")
    plt.close()

print("All figures generated and saved in the 'figures/' directory.")