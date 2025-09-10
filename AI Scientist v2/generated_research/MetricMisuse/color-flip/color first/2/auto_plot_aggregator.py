#!/usr/bin/env python3
"""
Final Aggregator Script for Scientific Figures
This script loads experiment .npy results from various experiments (baseline, research, and ablation)
and produces a final set of publishable figures saved in the "figures/" directory.
Each plotting block is wrapped in try-except so a failed plot does not stop the rest.
All figures use larger fonts (fontsize=12) for clarity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global font size for all plots
plt.rcParams.update({'font.size': 12})

# Create figures directory
os.makedirs("figures", exist_ok=True)

####################################
# 1. BASELINE: Embedding Dimension Tuning
####################################
try:
    # Load baseline experiment data from the exact file path
    baseline_path = "experiment_results/experiment_16b53856e1414051a86f1b52c8f17ae4_proc_1604462/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    # Navigate to the embedding_dim_tuning section on SPR_BENCH
    exp = baseline_data.get("embedding_dim_tuning", {}).get("SPR_BENCH", {})
    if exp:
        emb_dims = sorted([int(k.split("_")[-1]) for k in exp.keys()])
        train_losses = {}
        val_losses = {}
        test_metrics = {}
        for ed in emb_dims:
            key = f"emb_dim_{ed}"
            d = exp.get(key, {})
            train_losses[ed] = d.get("losses", {}).get("train", [])
            val_losses[ed] = d.get("losses", {}).get("val", [])
            test_metrics[ed] = d.get("metrics", {}).get("test", {})

        # Create one figure with 3 subplots: training loss, validation loss, and test metrics
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Subplot 1: Training Loss Curves
        for ed in emb_dims:
            epochs = range(1, len(train_losses[ed]) + 1)
            axs[0].plot(epochs, train_losses[ed], label=f"emb dim = {ed}")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Training Loss")
        axs[0].set_title("Baseline: Training Loss vs Epoch")
        axs[0].legend()
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)

        # Subplot 2: Validation Loss Curves
        for ed in emb_dims:
            epochs = range(1, len(val_losses[ed]) + 1)
            axs[1].plot(epochs, val_losses[ed], label=f"emb dim = {ed}")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Validation Loss")
        axs[1].set_title("Baseline: Validation Loss vs Epoch")
        axs[1].legend()
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)

        # Subplot 3: Test Metrics (Grouped Bar Chart)
        labels = ["CWA", "SWA", "GCWA"]
        x = np.arange(len(emb_dims))
        width = 0.25
        for i, metric in enumerate(labels):
            vals = [test_metrics[ed].get(metric, 0) for ed in emb_dims]
            axs[2].bar(x + (i - 1)*width, vals, width, label=metric)
        axs[2].set_xticks(x)
        axs[2].set_xticklabels([str(ed) for ed in emb_dims])
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel("Score")
        axs[2].set_title("Baseline: Test Metrics")
        axs[2].legend()
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)

        fig.suptitle("Baseline Experiment: Embedding Dimension Tuning", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "baseline_embedding_dimension.png"), dpi=300)
        plt.close(fig)
    else:
        print("Baseline experiment data not found.")
except Exception as e:
    print("Error in Baseline plotting:", e)
    plt.close('all')

####################################
# 2. RESEARCH: Enhanced BiLSTM + Clustering
####################################
try:
    research_path = "experiment_results/experiment_cbedf2ae312c480fa9f5304bff99b19b_proc_1608773/experiment_data.npy"
    research_data = np.load(research_path, allow_pickle=True).item()
    exp = research_data.get("SPR_BENCH", {})
    if exp:
        train_loss = exp.get("losses", {}).get("train", [])
        val_loss = exp.get("losses", {}).get("val", [])
        val_metrics = exp.get("metrics", {}).get("val", [])
        test_metrics = exp.get("metrics", {}).get("test", {})

        epochs = range(1, max(len(train_loss), len(val_loss)) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Subplot 1: Training vs Validation Loss
        if train_loss:
            axs[0].plot(epochs[:len(train_loss)], train_loss, label="Train")
        if val_loss:
            axs[0].plot(epochs[:len(val_loss)], val_loss, label="Validation")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Research: Loss Curves")
        axs[0].legend()
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)

        # Subplot 2: Validation Metrics (CWA, SWA, GCWA)
        cwa = [m.get("CWA", np.nan) for m in val_metrics] if val_metrics else []
        swa = [m.get("SWA", np.nan) for m in val_metrics] if val_metrics else []
        gcwa = [m.get("GCWA", np.nan) for m in val_metrics] if val_metrics else []
        axs[1].plot(epochs[:len(cwa)], cwa, label="CWA")
        axs[1].plot(epochs[:len(swa)], swa, label="SWA")
        axs[1].plot(epochs[:len(gcwa)], gcwa, label="GCWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Score")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("Research: Validation Metrics")
        axs[1].legend()
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)

        # Subplot 3: Test Metrics Bar Chart
        labels = ["CWA", "SWA", "GCWA"]
        vals = [test_metrics.get(l, 0) for l in labels]
        axs[2].bar(labels, vals, color=["steelblue", "orange", "green"])
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel("Score")
        axs[2].set_title("Research: Final Test Metrics")
        for i, v in enumerate(vals):
            axs[2].text(i, v + 0.02, f"{v:.2f}", ha="center")
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)

        fig.suptitle("Research Experiment: Enhanced BiLSTM + Clustering", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "research_enhanced_model.png"), dpi=300)
        plt.close(fig)
    else:
        print("Research experiment data not found.")
except Exception as e:
    print("Error in Research plotting:", e)
    plt.close('all')

####################################
# 3. ABLATION: No-Cluster-Embedding
####################################
try:
    no_cluster_path = "experiment_results/experiment_ab948d548e96430da8d9093f813939ea_proc_1614255/experiment_data.npy"
    no_cluster_data = np.load(no_cluster_path, allow_pickle=True).item()
    run = no_cluster_data.get("NoClusterEmbedding", {}).get("SPR_BENCH", {})
    if run:
        losses = run.get("losses", {})
        metrics = run.get("metrics", {})
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        val_metrics = metrics.get("val", [])
        test_metrics = metrics.get("test", {})

        epochs = range(1, max(len(train_loss), len(val_loss)) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curves
        axs[0].plot(epochs[:len(train_loss)], train_loss, label="Train")
        axs[0].plot(epochs[:len(val_loss)], val_loss, label="Validation")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("No-ClusterEmbedding: Loss Curves")
        axs[0].legend()

        # Validation metrics curves
        cwa = [m.get("CWA", np.nan) for m in val_metrics]
        swa = [m.get("SWA", np.nan) for m in val_metrics]
        gcwa = [m.get("GCWA", np.nan) for m in val_metrics]
        axs[1].plot(epochs[:len(cwa)], cwa, label="CWA")
        axs[1].plot(epochs[:len(swa)], swa, label="SWA")
        axs[1].plot(epochs[:len(gcwa)], gcwa, label="GCWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Metric Score")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("No-ClusterEmbedding: Validation Metrics")
        axs[1].legend()

        # Test metrics bar chart
        labels = list(test_metrics.keys())
        values = [test_metrics[k] for k in labels]
        axs[2].bar(labels, values, color="skyblue")
        axs[2].set_ylim(0, 1)
        axs[2].set_title("No-ClusterEmbedding: Test Metrics")
        for i, v in enumerate(values):
            axs[2].text(i, v+0.02, f"{v:.2f}", ha="center")
        fig.suptitle("Ablation: No-Cluster-Embedding", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "ablation_no_cluster_embedding.png"), dpi=300)
        plt.close(fig)
    else:
        print("No-Cluster-Embedding data not found.")
except Exception as e:
    print("Error in No-Cluster-Embedding plotting:", e)
    plt.close('all')

####################################
# 4. ABLATION: Multi-Synthetic-Dataset Training
####################################
try:
    multi_synth_path = "experiment_results/experiment_e63bd398362a49e5966bbbc4ab6baae2_proc_1614253/experiment_data.npy"
    multi_synth_data = np.load(multi_synth_path, allow_pickle=True).item()
    # multi_synth_data is expected to have keys for each synthetic dataset
    if multi_synth_data:
        datasets = list(multi_synth_data.keys())
        # Assume each dataset has same number of epochs for losses and metrics
        ep_count = len(next(iter(multi_synth_data.values())).get("losses", {}).get("train", []))
        epochs = list(range(1, ep_count + 1))
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Subplot 1: Loss curves for each dataset
        for name in datasets:
            rec = multi_synth_data[name]
            tr = rec.get("losses", {}).get("train", [])
            vl = rec.get("losses", {}).get("val", [])
            if tr and vl:
                axs[0].plot(epochs, tr, label=f"{name} - train")
                axs[0].plot(epochs, vl, label=f"{name} - val", linestyle="--")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Multi-Synth: Loss Curves")
        axs[0].legend()

        # Subplot 2: Validation Metrics over Epochs (for CWA, SWA, GCWA) grouped by dataset
        metrics_list = ["CWA", "SWA", "GCWA"]
        for m in metrics_list:
            for name in datasets:
                rec = multi_synth_data[name]
                val_mets = rec.get("metrics", {}).get("val", [])
                vals = [d.get(m, np.nan) for d in val_mets] if val_mets else []
                axs[1].plot(epochs[:len(vals)], vals, label=f"{name}-{m}")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Metric")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("Multi-Synth: Validation Metrics")
        axs[1].legend(fontsize=9)

        # Subplot 3: Test Metrics Bar Chart per dataset (grouped by metric)
        width = 0.25
        x = np.arange(len(datasets))
        for i, m in enumerate(metrics_list):
            vals = [multi_synth_data[name].get("metrics", {}).get("test", {}).get(m, np.nan) for name in datasets]
            axs[2].bar(x + (i - 1) * width, vals, width, label=m)
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(datasets)
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel("Score")
        axs[2].set_title("Multi-Synth: Test Metrics")
        axs[2].legend()
        fig.suptitle("Ablation: Multi-Synthetic-Dataset Training", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "ablation_multi_synth_training.png"), dpi=300)
        plt.close(fig)
    else:
        print("Multi-Synthetic-Dataset Training data not found.")
except Exception as e:
    print("Error in Multi-Synthetic-Dataset Training plotting:", e)
    plt.close('all')

####################################
# 5. ABLATION: Random-Cluster-Assignments
####################################
try:
    random_cluster_path = "experiment_results/experiment_0b75842fc3c84fd395336209aedffdba_proc_1614254/experiment_data.npy"
    random_cluster_data = np.load(random_cluster_path, allow_pickle=True).item()
    run = random_cluster_data.get("RandomCluster", {}).get("SPR_BENCH", {})
    if run:
        # Loss curves
        train_loss = run.get("losses", {}).get("train", [])
        val_loss = run.get("losses", {}).get("val", [])
        # Validation metrics
        val_metrics = run.get("metrics", {}).get("val", [])
        cwa = [m.get("CWA", np.nan) for m in val_metrics] if val_metrics else []
        swa = [m.get("SWA", np.nan) for m in val_metrics] if val_metrics else []
        gcwa = [m.get("GCWA", np.nan) for m in val_metrics] if val_metrics else []
        # Confusion matrix on test set using predictions and ground_truth
        preds = np.array(run.get("predictions", []))
        tgts = np.array(run.get("ground_truth", []))
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(train_loss) + 1)
        # Subplot 1: Loss curves
        axs[0].plot(epochs, train_loss, label="Train")
        axs[0].plot(epochs, val_loss, label="Validation")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("RandomCluster: Loss Curves")
        axs[0].legend()
        # Subplot 2: Validation metrics curves
        axs[1].plot(epochs[:len(cwa)], cwa, label="CWA")
        axs[1].plot(epochs[:len(swa)], swa, label="SWA")
        axs[1].plot(epochs[:len(gcwa)], gcwa, label="GCWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Metric")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("RandomCluster: Validation Metrics")
        axs[1].legend()
        # Subplot 3: Confusion Matrix (if predictions exist)
        if preds.size and tgts.size:
            num_classes = int(max(preds.max(), tgts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), int)
            for t, p in zip(tgts, preds):
                cm[int(t), int(p)] += 1
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")
            axs[2].set_title("RandomCluster: Confusion Matrix")
            for i in range(num_classes):
                for j in range(num_classes):
                    axs[2].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
        else:
            axs[2].text(0.5, 0.5, "No Confusion Data", ha="center")
        fig.suptitle("Ablation: Random-Cluster-Assignments", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "ablation_random_cluster.png"), dpi=300)
        plt.close(fig)
    else:
        print("Random-Cluster-Assignments data not found.")
except Exception as e:
    print("Error in Random-Cluster-Assignments plotting:", e)
    plt.close('all')

####################################
# 6. ABLATION: Uni-Directional-LSTM
####################################
try:
    uni_lstm_path = "experiment_results/experiment_17d508242e5946dab313da950afbe804_proc_1614257/experiment_data.npy"
    uni_lstm_data = np.load(uni_lstm_path, allow_pickle=True).item()
    run = uni_lstm_data.get("UniLSTM", {}).get("SPR_BENCH", {})
    if run:
        train_loss = run.get("losses", {}).get("train", [])
        val_loss = run.get("losses", {}).get("val", [])
        val_metrics = run.get("metrics", {}).get("val", [])
        cwa = [m.get("CWA", 0) for m in val_metrics]
        swa = [m.get("SWA", 0) for m in val_metrics]
        gcwa = [m.get("GCWA", 0) for m in val_metrics]
        preds = run.get("predictions", [])
        gts = run.get("ground_truth", [])
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(train_loss) + 1)
        # Loss curves
        axs[0].plot(epochs, train_loss, label="Train Loss")
        axs[0].plot(epochs, val_loss, label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("UniLSTM: Loss Curves")
        axs[0].legend()
        # Validation metrics curves
        axs[1].plot(epochs[:len(cwa)], cwa, label="CWA")
        axs[1].plot(epochs[:len(swa)], swa, label="SWA")
        axs[1].plot(epochs[:len(gcwa)], gcwa, label="GCWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Score")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("UniLSTM: Validation Metrics")
        axs[1].legend()
        # Confusion matrix
        if preds and gts:
            preds = np.array(preds)
            gts = np.array(gts)
            num_classes = int(len(set(gts)))
            cm = np.zeros((num_classes, num_classes), int)
            for t, p in zip(gts, preds):
                cm[int(t), int(p)] += 1
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")
            axs[2].set_title("UniLSTM: Confusion Matrix")
            for i in range(num_classes):
                for j in range(num_classes):
                    axs[2].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
        else:
            axs[2].text(0.5, 0.5, "No Confusion Data", ha="center")
        fig.suptitle("Ablation: Uni-Directional-LSTM", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "ablation_uni_lstm.png"), dpi=300)
        plt.close(fig)
    else:
        print("Uni-Directional-LSTM data not found.")
except Exception as e:
    print("Error in Uni-Directional-LSTM plotting:", e)
    plt.close('all')

####################################
# 7. ABLATION: Bag-of-Glyph-Pooling
####################################
try:
    bag_path = "experiment_results/experiment_0831521882aa40cfa04b9460db74cecc_proc_1614255/experiment_data.npy"
    bag_data = np.load(bag_path, allow_pickle=True).item()
    run = bag_data.get("BagOfGlyph", {}).get("SPR_BENCH", {})
    if run:
        tr_loss = run.get("losses", {}).get("train", [])
        val_loss = run.get("losses", {}).get("val", [])
        val_metrics = run.get("metrics", {}).get("val", [])
        cwa = [m.get("CWA", np.nan) for m in val_metrics]
        swa = [m.get("SWA", np.nan) for m in val_metrics]
        gcwa = [m.get("GCWA", np.nan) for m in val_metrics]
        preds = np.array(run.get("predictions", []))
        tgts = np.array(run.get("ground_truth", []))
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(tr_loss) + 1)
        # Loss curves
        axs[0].plot(epochs, tr_loss, label="Train")
        axs[0].plot(epochs, val_loss, label="Val")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Bag-of-Glyph: Loss Curves")
        axs[0].legend()
        # Validation metrics curves
        axs[1].plot(epochs[:len(cwa)], cwa, label="CWA")
        axs[1].plot(epochs[:len(swa)], swa, label="SWA")
        axs[1].plot(epochs[:len(gcwa)], gcwa, label="GCWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Score")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("Bag-of-Glyph: Validation Metrics")
        axs[1].legend()
        # Confusion Matrix
        if preds.size and tgts.size:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(tgts, preds, labels=sorted(set(tgts)))
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")
            axs[2].set_title("Bag-of-Glyph: Confusion Matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axs[2].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
        else:
            axs[2].text(0.5, 0.5, "No Confusion Data", ha="center")
        fig.suptitle("Ablation: Bag-of-Glyph-Pooling", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "ablation_bag_of_glyph.png"), dpi=300)
        plt.close(fig)
    else:
        print("Bag-of-Glyph-Pooling data not found.")
except Exception as e:
    print("Error in Bag-of-Glyph-Pooling plotting:", e)
    plt.close('all')

####################################
# 8. ABLATION: No-Color-Embedding
####################################
try:
    no_color_path = "experiment_results/experiment_add0e21d36da4a0483c0bf68b45f2a92_proc_1614257/experiment_data.npy"
    no_color_data = np.load(no_color_path, allow_pickle=True).item()
    # Use helper navigation for nested dict
    run = no_color_data.get("NoColorEmbedding", {}).get("SPR_BENCH", {})
    if run:
        loss_train = run.get("losses", {}).get("train", [])
        loss_val = run.get("losses", {}).get("val", [])
        metrics_val = run.get("metrics", {}).get("val", [])
        test_metrics = run.get("metrics", {}).get("test", {})
        preds = run.get("predictions", [])
        gts = run.get("ground_truth", [])
        epochs = range(1, len(loss_train) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        # Loss curves
        axs[0].plot(epochs, loss_train, label="Train Loss")
        axs[0].plot(epochs, loss_val, label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("No-ColorEmbedding: Loss Curves")
        axs[0].legend()
        # Validation metrics curves
        cwa = [m.get("CWA", np.nan) for m in metrics_val]
        swa = [m.get("SWA", np.nan) for m in metrics_val]
        gcwa = [m.get("GCWA", np.nan) for m in metrics_val]
        axs[1].plot(epochs[:len(cwa)], cwa, label="CWA")
        axs[1].plot(epochs[:len(swa)], swa, label="SWA")
        axs[1].plot(epochs[:len(gcwa)], gcwa, label="GCWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Score")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("No-ColorEmbedding: Validation Metrics")
        axs[1].legend()
        # Confusion Matrix (if available)
        if preds and gts:
            num_cls = int(max(max(preds), max(gts)) + 1)
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for p, t in zip(preds, gts):
                cm[int(t), int(p)] += 1
            im = axs[2].imshow(cm, cmap="Blues")
            axs[2].set_xlabel("Predicted")
            axs[2].set_ylabel("True")
            axs[2].set_title("No-ColorEmbedding: Confusion Matrix")
            for i in range(num_cls):
                for j in range(num_cls):
                    axs[2].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
        else:
            axs[2].text(0.5, 0.5, "No Confusion Data", ha="center")
        fig.suptitle("Ablation: No-Color-Embedding", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "ablation_no_color_embedding.png"), dpi=300)
        plt.close(fig)
    else:
        print("No-Color-Embedding data not found.")
except Exception as e:
    print("Error in No-Color-Embedding plotting:", e)
    plt.close('all')

####################################
# 9. ABLATION: Atomic-Glyph-Embedding (No Shape/Color Factorization)
####################################
try:
    atomic_path = "experiment_results/experiment_5b0bda42d4d34ff9b1ff6492adedc174_proc_1614254/experiment_data.npy"
    atomic_data = np.load(atomic_path, allow_pickle=True).item()
    run = atomic_data.get("AtomicGlyphEmbedding", {}).get("SPR_BENCH", {})
    if run:
        epochs = range(1, len(run.get("losses", {}).get("train", [])) + 1)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Loss curves
        axs[0].plot(epochs, run.get("losses", {}).get("train", []), label="Train Loss")
        axs[0].plot(epochs, run.get("losses", {}).get("val", []), label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("AtomicGlyphEmbedding: Loss Curves")
        axs[0].legend()
        # Validation metrics curves (CWA, SWA, GCWA)
        val_metrics = run.get("metrics", {}).get("val", [])
        for metric in ["CWA", "SWA", "GCWA"]:
            vals = [m.get(metric, np.nan) for m in val_metrics]
            axs[1].plot(epochs[:len(vals)], vals, label=metric)
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Metric")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("AtomicGlyphEmbedding: Validation Metrics")
        axs[1].legend()
        fig.suptitle("Ablation: Atomic-Glyph-Embedding", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "ablation_atomic_glyph_embedding.png"), dpi=300)
        plt.close(fig)
    else:
        print("Atomic-Glyph-Embedding data not found.")
except Exception as e:
    print("Error in Atomic-Glyph-Embedding plotting:", e)
    plt.close('all')

####################################
# 10. ABLATION: Sum-Fusion Embeddings (Shape + Color + Cluster)
####################################
try:
    sumfusion_path = "experiment_results/experiment_83fd22e9ba4944f8acba309b09e20baa_proc_1614253/experiment_data.npy"
    sumfusion_data = np.load(sumfusion_path, allow_pickle=True).item()
    # Assume that the Sum-Fusion results are stored under a key containing "SumFusion" (or similar)
    # For simplicity, iterate over keys and pick the one that appears related to Sum-Fusion.
    sf_key = None
    for key in sumfusion_data.keys():
        if "Sum" in key or "Fusion" in key:
            sf_key = key
            break
    if sf_key:
        rec = sumfusion_data[sf_key].get("SPR_BENCH", {})
        # Loss curves
        train_loss = rec.get("losses", {}).get("train", [])
        val_loss = rec.get("losses", {}).get("val", [])
        # Validation metric for CWA from list of dicts
        val_metrics = rec.get("metrics", {}).get("val", [])
        cwa_vals = [m.get("CWA", np.nan) for m in val_metrics] if val_metrics else []
        # Final test metrics (bar chart)
        test_metrics = rec.get("metrics", {}).get("test", {})

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        epochs_sf = range(1, len(train_loss) + 1)
        # Subplot 1: Loss curves
        axs[0].plot(epochs_sf, train_loss, label="Train Loss")
        axs[0].plot(epochs_sf, val_loss, label="Val Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Sum-Fusion: Loss Curves")
        axs[0].legend()
        # Subplot 2: Validation CWA Curve
        axs[1].plot(epochs_sf[:len(cwa_vals)], cwa_vals, marker="o", label="CWA")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Color-Weighted Accuracy")
        axs[1].set_ylim(0, 1)
        axs[1].set_title("Sum-Fusion: Validation CWA")
        axs[1].legend()
        # Subplot 3: Test Metrics Bar Chart
        labels = list(test_metrics.keys())
        values = [test_metrics[k] for k in labels]
        axs[2].bar(labels, values, color="skyblue")
        axs[2].set_ylim(0, 1)
        axs[2].set_title("Sum-Fusion: Test Metrics")
        for i, v in enumerate(values):
            axs[2].text(i, v + 0.02, f"{v:.2f}", ha="center")
        fig.suptitle("Ablation: Sum-Fusion Embeddings", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join("figures", "ablation_sum_fusion_embeddings.png"), dpi=300)
        plt.close(fig)
    else:
        print("Sum-Fusion Embeddings key not found in the data.")
except Exception as e:
    print("Error in Sum-Fusion Embeddings plotting:", e)
    plt.close('all')

print("All plots generated and saved in the 'figures/' directory.")