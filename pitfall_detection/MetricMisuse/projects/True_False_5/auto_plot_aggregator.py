"""
Final Aggregator Script for Zero-Shot Synthetic PolyRule Reasoning Results

This script loads the .npy experiment results produced in the baseline,
research, and ablation studies and generates publication‐quality figures
saved in the "figures/" directory. Each plot is wrapped in its own try/except
block so that one failure does not prevent the remainder from executing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global plotting style for clarity in publication
plt.rcParams.update({
    'font.size': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
os.makedirs("figures", exist_ok=True)

# ----------------------------------------------------------------
# File paths (use the exact full paths from the JSON summaries)
baseline_file = "experiment_results/experiment_1191d2f2e9884009aa1dda0d5ea5ad3c_proc_2676156/experiment_data.npy"
research_file = "experiment_results/experiment_107634f1a9f24333ab304c876afd3618_proc_2678329/experiment_data.npy"
no_symbolic_file = "experiment_results/experiment_a6831f04c1cb451aaa0fde30ad5c8053_proc_2682348/experiment_data.npy"
no_position_file = "experiment_results/experiment_bb14a2fd3aef48d1a6daa75810b2528f_proc_2682349/experiment_data.npy"
no_color_file = "experiment_results/experiment_f428dfe82c7a4eaea5bd91f6a9ace3ed_proc_2682350/experiment_data.npy"
no_transformer_file = "experiment_results/experiment_497b500b3d2b4e41aad4921032cf4cd9_proc_2682351/experiment_data.npy"
multi_dataset_file = "experiment_results/experiment_962cc39d034c4ac5af28eaea24e6381c_proc_2682348/experiment_data.npy"
symbolic_only_file = "experiment_results/experiment_bcd2d8fdc6e94348bd5a4b833eea2f3b_proc_2682351/experiment_data.npy"
no_shape_file = "experiment_results/experiment_d749d41979734473a0dfc1a1a7986c31_proc_2682349/experiment_data.npy"
shuffled_sym_file = "experiment_results/experiment_5664eac8f8e845a8abbef1087a9196c0_proc_2682350/experiment_data.npy"

# Utility: remove extra whitespace from a figure and close axes.
def tidy_up():
    plt.tight_layout()
    for ax in plt.gcf().get_axes():
        ax.label_outer()

# ==============================
# Figure 1: Baseline Results
#   (a) Loss and RCWA curves combined into one figure with 2 subplots.
#   (b) Test accuracy bar plot.
# ==============================
try:
    data = np.load(baseline_file, allow_pickle=True).item()
    # Assume baseline file contains a record under key "SPR_BENCH"
    record = data.get("SPR_BENCH", {})
    epochs = np.arange(1, len(record.get("losses", {}).get("train", [])) + 1)
    train_loss = np.array(record.get("losses", {}).get("train", []))
    val_loss = np.array(record.get("losses", {}).get("val", []))
    train_rcwa = np.array(record.get("metrics", {}).get("train_rcwa", []))
    val_rcwa = np.array(record.get("metrics", {}).get("val_rcwa", []))
    preds = np.array(record.get("predictions", []))
    gts = np.array(record.get("ground_truth", []))
    
    # Plot (a): Loss and RCWA curves
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axs[0].plot(epochs, train_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].set_title("Baseline: Loss Curves")
    axs[0].legend()
    
    axs[1].plot(epochs, train_rcwa, label="Train RCWA")
    axs[1].plot(epochs, val_rcwa, label="Validation RCWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("RCWA")
    axs[1].set_title("Baseline: RCWA Curves")
    axs[1].legend()
    
    tidy_up()
    fig.savefig(os.path.join("figures", "baseline_loss_rcwa.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Baseline Loss/RCWA plot: {e}")

try:
    # Plot (b): Test accuracy as bar plot for baseline
    if preds.size and gts.size:
        test_acc = (preds == gts).mean()
    else:
        test_acc = 0
    plt.figure(figsize=(5, 4), dpi=300)
    plt.bar(["Test Accuracy"], [test_acc], color="steelblue")
    plt.ylim(0, 1)
    plt.title("Baseline: Test Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error in Baseline Test Accuracy plot: {e}")

# ==============================
# Figure 2: Research Results
#   (a) Loss and SWA curves combined.
#   (b) Test accuracy bar plot.
# ==============================
try:
    data = np.load(research_file, allow_pickle=True).item()
    # Assume research file has key "SPR_BENCH"; using SWA here.
    record = data.get("SPR_BENCH", {})
    epochs = np.arange(1, len(record.get("losses", {}).get("train", [])) + 1)
    train_loss = np.array(record.get("losses", {}).get("train", []))
    val_loss = np.array(record.get("losses", {}).get("val", []))
    train_swa = np.array(record.get("metrics", {}).get("train_swa", []))
    val_swa = np.array(record.get("metrics", {}).get("val_swa", []))
    preds = np.array(record.get("predictions", []))
    gts = np.array(record.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axs[0].plot(epochs, train_loss, label="Train")
    axs[0].plot(epochs, val_loss, label="Validation")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-Entropy Loss")
    axs[0].set_title("Research: Loss Curves")
    axs[0].legend()

    axs[1].plot(epochs, train_swa, label="Train SWA")
    axs[1].plot(epochs, val_swa, label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Shape-Weighted Accuracy")
    axs[1].set_title("Research: SWA Curves")
    axs[1].legend()
    
    tidy_up()
    fig.savefig(os.path.join("figures", "research_loss_swa.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Research Loss/SWA plot: {e}")

try:
    # Research: Test Accuracy Bar Plot
    if preds.size and gts.size:
        test_acc = (preds == gts).mean()
    else:
        test_acc = 0
    plt.figure(figsize=(5,4), dpi=300)
    plt.bar(["Test Accuracy"], [test_acc], color="darkorange")
    plt.ylim(0, 1)
    plt.title("Research: Test Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "research_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error in Research Test Accuracy plot: {e}")

# ==============================
# Figure 3: Ablation - No-Symbolic-Vector
# Combine loss curve, SWA curve, and confusion matrix in 3 subplots.
# ==============================
try:
    data = np.load(no_symbolic_file, allow_pickle=True).item()
    # Record under key "NoSymbolicVector" then "SPR_BENCH"
    record = data.get("NoSymbolicVector", {}).get("SPR_BENCH", {})
    epochs = np.arange(1, len(record.get("losses", {}).get("train", [])) + 1)
    train_loss = np.array(record.get("losses", {}).get("train", []))
    val_loss = np.array(record.get("losses", {}).get("val", []))
    train_swa = np.array(record.get("metrics", {}).get("train_swa", []))
    val_swa = np.array(record.get("metrics", {}).get("val_swa", []))
    preds = np.array(record.get("predictions", []))
    gts = np.array(record.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    # Loss curves
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("No-Symbolic-Vector: Loss")
    axs[0].legend()
    # SWA curves
    axs[1].plot(epochs, train_swa, label="Train SWA")
    axs[1].plot(epochs, val_swa, label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("No-Symbolic-Vector: SWA")
    axs[1].legend()
    # Confusion matrix
    if preds.size and gts.size:
        n_cls = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_title("No-Symbolic-Vector: Confusion Matrix")
        plt.colorbar(im, ax=axs[2])
    else:
        axs[2].text(0.5, 0.5, "No data", ha="center")
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_no_symbolic_vector.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in No-Symbolic-Vector ablation plot: {e}")

# ==============================
# Figure 4: Ablation - No-Position-Embedding
# Combine loss, SWA and confusion matrix in one figure with 3 subplots.
# ==============================
try:
    data = np.load(no_position_file, allow_pickle=True).item()
    # Record under key "no_position_embedding" then "SPR_BENCH"
    record = data.get("no_position_embedding", {}).get("SPR_BENCH", {})
    epochs = np.arange(1, len(record.get("losses", {}).get("train", [])) + 1)
    train_loss = np.array(record.get("losses", {}).get("train", []))
    val_loss = np.array(record.get("losses", {}).get("val", []))
    train_swa = np.array(record.get("metrics", {}).get("train_swa", []))
    val_swa = np.array(record.get("metrics", {}).get("val_swa", []))
    preds = np.array(record.get("predictions", []))
    gts = np.array(record.get("ground_truth", []))

    fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("No-Position-Embedding: Loss")
    axs[0].legend()
    
    axs[1].plot(epochs, train_swa, label="Train SWA")
    axs[1].plot(epochs, val_swa, label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("No-Position-Embedding: SWA")
    axs[1].legend()
    
    if preds.size and gts.size:
        n_cls = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_title("No-Position-Embedding: Confusion")
        plt.colorbar(im, ax=axs[2])
    else:
        axs[2].text(0.5, 0.5, "No data", ha="center")
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_no_position_embedding.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in No-Position-Embedding ablation plot: {e}")

# ==============================
# Figure 5: Ablation - No-Color-Embedding
# Combine loss, SWA and confusion matrix.
# ==============================
try:
    data = np.load(no_color_file, allow_pickle=True).item()
    # Record under key "NoColorEmbedding" then "SPR_BENCH"
    record = data.get("NoColorEmbedding", {}).get("SPR_BENCH", {})
    epochs = np.arange(1, len(record.get("losses", {}).get("train", [])) + 1)
    train_loss = np.array(record.get("losses", {}).get("train", []))
    val_loss = np.array(record.get("losses", {}).get("val", []))
    train_swa = np.array(record.get("metrics", {}).get("train_swa", []))
    val_swa = np.array(record.get("metrics", {}).get("val_swa", []))
    preds = np.array(record.get("predictions", []))
    gts = np.array(record.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("No-Color-Embedding: Loss")
    axs[0].legend()
    
    axs[1].plot(epochs, train_swa, label="Train SWA")
    axs[1].plot(epochs, val_swa, label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("No-Color-Embedding: SWA")
    axs[1].legend()
    
    if preds.size and gts.size:
        n_cls = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_title("No-Color-Embedding: Confusion")
        plt.colorbar(im, ax=axs[2])
    else:
        axs[2].text(0.5, 0.5, "No data", ha="center")
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_no_color_embedding.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in No-Color-Embedding ablation plot: {e}")

# ==============================
# Figure 6: Ablation - No-Transformer-Encoder (Bag-of-Embeddings)
# Combine loss, SWA and confusion matrix.
# ==============================
try:
    data = np.load(no_transformer_file, allow_pickle=True).item()
    # Record under key "NoTransformerEncoder" then "SPR_BENCH"
    record = data.get("NoTransformerEncoder", {}).get("SPR_BENCH", {})
    epochs = np.arange(1, len(record.get("losses", {}).get("train", [])) + 1)
    train_loss = np.array(record.get("losses", {}).get("train", []))
    val_loss = np.array(record.get("losses", {}).get("val", []))
    train_swa = np.array(record.get("metrics", {}).get("train_swa", []))
    val_swa = np.array(record.get("metrics", {}).get("val_swa", []))
    preds = np.array(record.get("predictions", []))
    gts = np.array(record.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("No-Transformer-Encoder: Loss")
    axs[0].legend()
    
    axs[1].plot(epochs, train_swa, label="Train SWA")
    axs[1].plot(epochs, val_swa, label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("No-Transformer-Encoder: SWA")
    axs[1].legend()
    
    if preds.size and gts.size:
        n_cls = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_title("No-Transformer-Encoder: Confusion")
        plt.colorbar(im, ax=axs[2])
    else:
        axs[2].text(0.5, 0.5, "No data", ha="center")
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_no_transformer_encoder.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in No-Transformer-Encoder ablation plot: {e}")

# ==============================
# Figure 7: Ablation - Multi-Synthetic-Dataset Training
# Create combined figure: loss curves, SWA curves, and test accuracy bar.
# ==============================
try:
    data = np.load(multi_dataset_file, allow_pickle=True).item()
    # Record is under key "multi_dataset", a dict of datasets.
    multi = data.get("multi_dataset", {})
    fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    # Loss curves aggregated
    for name, rec in multi.items():
        ep = np.arange(1, len(rec.get("losses", {}).get("train", [])) + 1)
        axs[0].plot(ep, rec["losses"]["train"], label=f"{name} Train")
        axs[0].plot(ep, rec["losses"]["val"], label=f"{name} Val", linestyle="--")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Multi-Dataset: Loss Curves")
    axs[0].legend(fontsize=10)
    # SWA curves aggregated
    for name, rec in multi.items():
        ep = np.arange(1, len(rec.get("metrics", {}).get("train_swa", [])) + 1)
        axs[1].plot(ep, rec["metrics"]["train_swa"], label=f"{name} Train")
        axs[1].plot(ep, rec["metrics"]["val_swa"], label=f"{name} Val", linestyle="--")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("Multi-Dataset: SWA Curves")
    axs[1].legend(fontsize=10)
    # Test accuracy bar: average accuracy per dataset
    names, accs = [], []
    for name, rec in multi.items():
        p = np.array(rec.get("predictions", []))
        g = np.array(rec.get("ground_truth", []))
        if p.size and g.size:
            names.append(name)
            accs.append((p==g).mean())
    if names:
        axs[2].bar(names, accs, color="seagreen")
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel("Accuracy")
        axs[2].set_title("Multi-Dataset: Test Accuracy")
    else:
        axs[2].text(0.5, 0.5, "No data", ha="center")
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_multi_dataset.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Multi-Synthetic-Dataset Training plot: {e}")

# ==============================
# Figure 8: Ablation - Symbolic-Only (Remove All Token Embeddings)
# Combine loss, SWA and confusion matrix.
# ==============================
try:
    data = np.load(symbolic_only_file, allow_pickle=True).item()
    record = data.get("SYM_ONLY", {}).get("SPR_BENCH", {})
    epochs = np.arange(1, len(record.get("losses", {}).get("train", [])) + 1)
    train_loss = np.array(record.get("losses", {}).get("train", []))
    val_loss = np.array(record.get("losses", {}).get("val", []))
    train_swa = np.array(record.get("metrics", {}).get("train_swa", []))
    val_swa = np.array(record.get("metrics", {}).get("val_swa", []))
    preds = np.array(record.get("predictions", []))
    gts = np.array(record.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Symbolic-Only: Loss")
    axs[0].legend()
    
    axs[1].plot(epochs, train_swa, label="Train SWA")
    axs[1].plot(epochs, val_swa, label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("Symbolic-Only: SWA")
    axs[1].legend()
    
    if preds.size and gts.size:
        n_cls = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_title("Symbolic-Only: Confusion")
        plt.colorbar(im, ax=axs[2])
    else:
        axs[2].text(0.5, 0.5, "No data", ha="center")
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_symbolic_only.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Symbolic-Only ablation plot: {e}")

# ==============================
# Figure 9: Ablation - No-Shape-Embedding (Color-Only Tokens)
# Combined figure with loss, SWA and confusion matrix.
# ==============================
try:
    data = np.load(no_shape_file, allow_pickle=True).item()
    # Use a helper that gets the single record; assume structure with one key.
    model_key = next(iter(data)) if data else None
    record = data.get(model_key, {}) if model_key else {}
    # Assume dataset key is present (like "SPR_BENCH")
    dset = next(iter(record)) if record else None
    rec = record.get(dset, {})
    epochs = np.arange(1, len(rec.get("losses", {}).get("train", [])) + 1)
    train_loss = np.array(rec.get("losses", {}).get("train", []))
    val_loss = np.array(rec.get("losses", {}).get("val", []))
    train_swa = np.array(rec.get("metrics", {}).get("train_swa", []))
    val_swa = np.array(rec.get("metrics", {}).get("val_swa", []))
    preds = np.array(rec.get("predictions", []))
    gts = np.array(rec.get("ground_truth", []))
    
    fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title(f"{dset} Loss (Color-Only)")
    axs[0].legend()
    
    axs[1].plot(epochs, train_swa, label="Train SWA")
    axs[1].plot(epochs, val_swa, label="Validation SWA")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title(f"{dset} SWA (Color-Only)")
    axs[1].legend()
    
    if preds.size and gts.size:
        n_cls = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        im = axs[2].imshow(cm, cmap="Blues")
        axs[2].set_xlabel("Predicted")
        axs[2].set_ylabel("True")
        axs[2].set_title(f"{dset} Confusion (Color-Only)")
        plt.colorbar(im, ax=axs[2])
    else:
        axs[2].text(0.5, 0.5, "No data", ha="center")
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_no_shape_embedding.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in No-Shape-Embedding ablation plot: {e}")

# ==============================
# Figure 10: Ablation - Shuffled-Symbolic-Vector
# Create one figure for loss and SWA curves; and a second figure for confusion matrices.
# ==============================
try:
    data = np.load(shuffled_sym_file, allow_pickle=True).item()
    variants = ["ORIG_SYM", "SHUFFLED_SYM"]
    dataset = "SPR_BENCH"
    # Figure 10A: Loss and SWA curves in one figure with 2 subplots.
    fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=300)
    for var in variants:
        rec = data.get(var, {}).get(dataset, {})
        if rec:
            ep = np.arange(1, len(rec.get("losses", {}).get("train", [])) + 1)
            axs[0].plot(ep, rec["losses"]["train"], label=f"{var} Train")
            axs[0].plot(ep, rec["losses"]["val"], label=f"{var} Val", linestyle="--")
            ep2 = np.arange(1, len(rec.get("metrics", {}).get("train_swa", [])) + 1)
            axs[1].plot(ep2, rec["metrics"]["train_swa"], label=f"{var} Train")
            axs[1].plot(ep2, rec["metrics"]["val_swa"], label=f"{var} Val", linestyle="--")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Shuffled-Symbolic: Loss Curves")
    axs[0].legend()
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("SWA")
    axs[1].set_title("Shuffled-Symbolic: SWA Curves")
    axs[1].legend()
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_shuffled_sym_loss_swa.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Shuffled-Symbolic loss/SWA plot: {e}")

try:
    # Figure 10B: Confusion matrices for each variant in side-by-side subplots.
    fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=300)
    for i, var in enumerate(variants):
        rec = data.get(var, {}).get(dataset, {})
        if rec:
            preds = np.array(rec.get("predictions", []))
            gts = np.array(rec.get("ground_truth", []))
            if preds.size and gts.size:
                n_cls = int(max(gts.max(), preds.max())) + 1
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                im = axs[i].imshow(cm, cmap="Blues")
                axs[i].set_xlabel("Predicted")
                axs[i].set_ylabel("True")
                axs[i].set_title(f"{var} Confusion")
                plt.colorbar(im, ax=axs[i])
            else:
                axs[i].text(0.5, 0.5, "No data", ha="center")
    tidy_up()
    fig.savefig(os.path.join("figures", "ablation_shuffled_sym_confusion.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Shuffled-Symbolic confusion matrix plot: {e}")

print("All final figures have been saved in the 'figures/' directory.")