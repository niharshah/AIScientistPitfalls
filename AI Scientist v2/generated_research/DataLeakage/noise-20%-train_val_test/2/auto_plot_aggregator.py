#!/usr/bin/env python3
"""
Final Results Aggregator for Contextual Embedding-Based SPR Research

This script loads experiment results from pre-saved .npy files for both
the Baseline and Research experiments and produces the final set of publishable figures.
All figures are saved under the "figures/" directory.
Each figure is wrapped in a try-except block so that failure in one does not
prevent the remaining plots from being generated.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Increase font size for publication-quality figures
plt.rcParams.update({'font.size': 14})

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Utility to remove top/right spines for a cleaner look
def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Utility to load .npy experiment data
def load_data(file_path):
    return np.load(file_path, allow_pickle=True).item()

def main():
    # -------------------------------------------------------------------------
    # Load Experiment Data from full, exact file paths provided in summaries
    # -------------------------------------------------------------------------
    try:
        baseline_data = load_data(
            "experiment_results/experiment_e323339a7c5841b299bfd57947090433_proc_3158726/experiment_data.npy"
        )
    except Exception as e:
        print("Error loading Baseline data:", e)
        baseline_data = {}

    try:
        research_data = load_data(
            "experiment_results/experiment_85de7cecb4ed48f7907db63927146615_proc_3164419/experiment_data.npy"
        )
    except Exception as e:
        print("Error loading Research data:", e)
        research_data = {}

    # -------------------------------------------------------------------------
    # Figure 1: Aggregated Macro-F1 Curves (Side by Side: Baseline & Research)
    # -------------------------------------------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Baseline: expect data under "num_epochs"
        if "num_epochs" in baseline_data:
            keys = list(baseline_data["num_epochs"].keys())
            colors = plt.cm.tab10.colors
            ax = axes[0]
            for idx, k in enumerate(keys):
                entry = baseline_data["num_epochs"][k]
                epochs = entry.get("epochs", [])
                train_f1 = entry.get("metrics", {}).get("train_macro_f1", [])
                val_f1 = entry.get("metrics", {}).get("val_macro_f1", [])
                ax.plot(epochs, train_f1, linestyle="--", color=colors[idx % len(colors)], label=f"{k} train")
                ax.plot(epochs, val_f1, linestyle="-", color=colors[idx % len(colors)], label=f"{k} val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro F1")
            ax.set_title("Baseline Macro F1 Curves")
            ax.legend()
            remove_spines(ax)
        else:
            axes[0].text(0.5, 0.5, "No Baseline Macro F1 Data", ha="center", va="center")
        
        # Research: iterate over top-level keys
        keys = list(research_data.keys())
        colors = plt.cm.tab10.colors
        ax = axes[1]
        for idx, k in enumerate(keys):
            entry = research_data[k]
            epochs = entry.get("epochs", [])
            train_f1 = entry.get("metrics", {}).get("train_macro_f1", [])
            val_f1 = entry.get("metrics", {}).get("val_macro_f1", [])
            if epochs:
                ax.plot(epochs, train_f1, linestyle="--", color=colors[idx % len(colors)], label=f"{k} train")
                ax.plot(epochs, val_f1, linestyle="-", color=colors[idx % len(colors)], label=f"{k} val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.set_title("Research Macro F1 Curves")
        ax.legend()
        remove_spines(ax)
        plt.tight_layout()
        fig.savefig("figures/aggregated_macro_f1.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        print("Error creating aggregated Macro-F1 plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: Aggregated Loss Curves (Side by Side: Baseline & Research)
    # -------------------------------------------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Baseline Loss Curves
        if "num_epochs" in baseline_data:
            keys = list(baseline_data["num_epochs"].keys())
            colors = plt.cm.tab10.colors
            ax = axes[0]
            for idx, k in enumerate(keys):
                entry = baseline_data["num_epochs"][k]
                epochs = entry.get("epochs", [])
                train_loss = entry.get("losses", {}).get("train", [])
                val_loss = entry.get("losses", {}).get("val", [])
                ax.plot(epochs, train_loss, linestyle="--", color=colors[idx % len(colors)], label=f"{k} train")
                ax.plot(epochs, val_loss, linestyle="-", color=colors[idx % len(colors)], label=f"{k} val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Baseline Loss Curves")
            ax.legend()
            remove_spines(ax)
        else:
            axes[0].text(0.5, 0.5, "No Baseline Loss Data", ha="center", va="center")
        
        # Research Loss Curves
        keys = list(research_data.keys())
        colors = plt.cm.tab10.colors
        ax = axes[1]
        for idx, k in enumerate(keys):
            entry = research_data[k]
            epochs = entry.get("epochs", [])
            train_loss = entry.get("losses", {}).get("train", [])
            val_loss = entry.get("losses", {}).get("val", [])
            if epochs:
                ax.plot(epochs, train_loss, linestyle="--", color=colors[idx % len(colors)], label=f"{k} train")
                ax.plot(epochs, val_loss, linestyle="-", color=colors[idx % len(colors)], label=f"{k} val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Research Loss Curves")
        ax.legend()
        remove_spines(ax)
        plt.tight_layout()
        fig.savefig("figures/aggregated_loss_curves.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        print("Error creating aggregated Loss Curve plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 3: Test Macro-F1 Comparison Bar Chart (Baseline and Research)
    # -------------------------------------------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Baseline Test Macro-F1 Bar Chart
        baseline_scores = {}
        if "num_epochs" in baseline_data:
            for k, entry in baseline_data["num_epochs"].items():
                baseline_scores[k] = entry.get("test_macro_f1", np.nan)
            ax = axes[0]
            ax.bar(list(baseline_scores.keys()), list(baseline_scores.values()), color='skyblue')
            ax.set_xlabel("Hyperparameter Setting")
            ax.set_ylabel("Test Macro F1")
            ax.set_title("Baseline Test Macro F1")
            ax.set_xticklabels(list(baseline_scores.keys()), rotation=45)
            remove_spines(ax)
        else:
            axes[0].text(0.5, 0.5, "No Baseline Test Data", ha="center", va="center")
        
        # Research Test Macro-F1 Bar Chart
        research_scores = {}
        for k, entry in research_data.items():
            research_scores[k] = entry.get("test_macro_f1", np.nan)
        ax = axes[1]
        if research_scores:
            ax.bar(list(research_scores.keys()), list(research_scores.values()), color='orchid')
            ax.set_xlabel("Experiment Key")
            ax.set_ylabel("Test Macro F1")
            ax.set_title("Research Test Macro F1")
            ax.set_xticklabels(list(research_scores.keys()), rotation=45)
            remove_spines(ax)
        else:
            axes[1].text(0.5, 0.5, "No Research Test Data", ha="center", va="center")
        plt.tight_layout()
        fig.savefig("figures/test_macro_f1_comparison.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        print("Error creating Test Macro-F1 Comparison plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 4: Research Confusion Matrices (Multiple Subplots in a Grid)
    # -------------------------------------------------------------------------
    try:
        # Only include keys where both predictions and ground truth exist
        research_keys_with_cm = [
            k for k, entry in research_data.items() 
            if entry.get("predictions") is not None and entry.get("ground_truth") is not None
        ]
        if research_keys_with_cm:
            n = len(research_keys_with_cm)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            # Ensure axes is a 2D array
            if rows == 1:
                axes = np.array(axes).reshape(1, -1)
            for idx, k in enumerate(research_keys_with_cm):
                entry = research_data[k]
                preds = entry.get("predictions")
                gts = entry.get("ground_truth")
                cm_matrix = confusion_matrix(gts, preds)
                ax = axes[idx // cols, idx % cols]
                im = ax.imshow(cm_matrix, cmap="Blues")
                ax.set_title(f"{k} Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                remove_spines(ax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Hide any unused subplots
            total_subplots = rows * cols
            for j in range(idx + 1, total_subplots):
                ax = axes[j // cols, j % cols]
                ax.axis('off')
            plt.tight_layout()
            fig.savefig("figures/research_confusion_matrices.png", dpi=300)
            plt.close(fig)
        else:
            print("No research confusion matrix data available.")
    except Exception as e:
        print("Error creating Research Confusion Matrices plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 5: Baseline Test Macro-F1 vs. Number of Epochs (Line Plot)
    # -------------------------------------------------------------------------
    try:
        if "num_epochs" in baseline_data:
            settings = []
            test_scores = []
            for k, entry in baseline_data["num_epochs"].items():
                try:
                    # Expect key format like "epochs_5" -> extract numeric value 5
                    epoch_num = int(k.split("_")[1])
                except Exception:
                    epoch_num = 0
                settings.append(epoch_num)
                test_scores.append(entry.get("test_macro_f1", np.nan))
            # Sort by number of epochs
            sorted_pairs = sorted(zip(settings, test_scores), key=lambda x: x[0])
            epochs_sorted, scores_sorted = zip(*sorted_pairs)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(epochs_sorted, scores_sorted, marker="o", linestyle="-", color="green")
            ax.set_xlabel("Number of Epochs")
            ax.set_ylabel("Test Macro F1")
            ax.set_title("Baseline: Test Macro F1 vs. Number of Epochs")
            remove_spines(ax)
            plt.tight_layout()
            fig.savefig("figures/baseline_test_macro_f1_vs_epochs.png", dpi=300)
            plt.close(fig)
        else:
            print("No baseline epoch data for Test Macro-F1 vs. Epochs plot.")
    except Exception as e:
        print("Error creating Baseline Test Macro-F1 vs. Epochs plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 6: Research Average Macro-F1 Curves (Across Experiment Keys)
    # -------------------------------------------------------------------------
    try:
        keys = list(research_data.keys())
        if keys:
            avg_train = None
            avg_val = None
            count = 0
            epoch_range = None
            for k in keys:
                entry = research_data[k]
                epochs = entry.get("epochs", [])
                if not epochs:
                    continue
                if epoch_range is None:
                    epoch_range = epochs
                train = np.array(entry.get("metrics", {}).get("train_macro_f1", []))
                val = np.array(entry.get("metrics", {}).get("val_macro_f1", []))
                if avg_train is None:
                    avg_train = train
                    avg_val = val
                else:
                    avg_train += train
                    avg_val += val
                count += 1
            if count > 0:
                avg_train /= count
                avg_val /= count
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(epoch_range, avg_train, linestyle="--", marker="o", label="Avg Train Macro F1")
                ax.plot(epoch_range, avg_val, linestyle="-", marker="o", label="Avg Val Macro F1")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Macro F1")
                ax.set_title("Research Average Macro F1 Curves")
                ax.legend()
                remove_spines(ax)
                plt.tight_layout()
                fig.savefig("figures/research_avg_macro_f1.png", dpi=300)
                plt.close(fig)
            else:
                print("No valid research data for average Macro F1 plot.")
        else:
            print("No research keys for average Macro F1 plot.")
    except Exception as e:
        print("Error creating Research Average Macro-F1 plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 7: Research Average Loss Curves (Across Experiment Keys)
    # -------------------------------------------------------------------------
    try:
        keys = list(research_data.keys())
        if keys:
            avg_train_loss = None
            avg_val_loss = None
            count = 0
            epoch_range = None
            for k in keys:
                entry = research_data[k]
                epochs = entry.get("epochs", [])
                if not epochs:
                    continue
                if epoch_range is None:
                    epoch_range = epochs
                train_loss = np.array(entry.get("losses", {}).get("train", []))
                val_loss = np.array(entry.get("losses", {}).get("val", []))
                if avg_train_loss is None:
                    avg_train_loss = train_loss
                    avg_val_loss = val_loss
                else:
                    avg_train_loss += train_loss
                    avg_val_loss += val_loss
                count += 1
            if count > 0:
                avg_train_loss /= count
                avg_val_loss /= count
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(epoch_range, avg_train_loss, linestyle="--", marker="o", label="Avg Train Loss")
                ax.plot(epoch_range, avg_val_loss, linestyle="-", marker="o", label="Avg Val Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Research Average Loss Curves")
                ax.legend()
                remove_spines(ax)
                plt.tight_layout()
                fig.savefig("figures/research_avg_loss.png", dpi=300)
                plt.close(fig)
            else:
                print("No valid research data for average Loss plot.")
        else:
            print("No research keys for average Loss plot.")
    except Exception as e:
        print("Error creating Research Average Loss plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 8: Detailed Baseline Macro-F1 for Setting 'epochs_10'
    # -------------------------------------------------------------------------
    try:
        if "num_epochs" in baseline_data and "epochs_10" in baseline_data["num_epochs"]:
            entry = baseline_data["num_epochs"]["epochs_10"]
            epochs = entry.get("epochs", [])
            train_f1 = entry.get("metrics", {}).get("train_macro_f1", [])
            val_f1 = entry.get("metrics", {}).get("val_macro_f1", [])
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(epochs, train_f1, linestyle="--", marker="o", label="Train Macro F1")
            ax.plot(epochs, val_f1, linestyle="-", marker="o", label="Val Macro F1")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro F1")
            ax.set_title("Baseline 'epochs 10' Macro F1 Detail")
            ax.legend()
            remove_spines(ax)
            plt.tight_layout()
            fig.savefig("figures/baseline_epochs_10_macro_f1_detail.png", dpi=300)
            plt.close(fig)
        else:
            print("Baseline 'epochs_10' data not available for detailed Macro F1 plot.")
    except Exception as e:
        print("Error creating Baseline epochs_10 Macro-F1 detail plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 9: Detailed Baseline Loss for Setting 'epochs_30'
    # -------------------------------------------------------------------------
    try:
        if "num_epochs" in baseline_data and "epochs_30" in baseline_data["num_epochs"]:
            entry = baseline_data["num_epochs"]["epochs_30"]
            epochs = entry.get("epochs", [])
            train_loss = entry.get("losses", {}).get("train", [])
            val_loss = entry.get("losses", {}).get("val", [])
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(epochs, train_loss, linestyle="--", marker="o", label="Train Loss")
            ax.plot(epochs, val_loss, linestyle="-", marker="o", label="Val Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Baseline 'epochs 30' Loss Detail")
            ax.legend()
            remove_spines(ax)
            plt.tight_layout()
            fig.savefig("figures/baseline_epochs_30_loss_detail.png", dpi=300)
            plt.close(fig)
        else:
            print("Baseline 'epochs_30' data not available for detailed Loss plot.")
    except Exception as e:
        print("Error creating Baseline epochs_30 Loss detail plot:", e)
        plt.close()

    # -------------------------------------------------------------------------
    # Figure 10: Detailed Research Confusion Matrix for the First Experiment Key
    # -------------------------------------------------------------------------
    try:
        research_keys = list(research_data.keys())
        if research_keys:
            key0 = research_keys[0]
            entry = research_data[key0]
            preds = entry.get("predictions")
            gts = entry.get("ground_truth")
            if preds is not None and gts is not None:
                cm_matrix = confusion_matrix(gts, preds)
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.imshow(cm_matrix, cmap="Blues")
                ax.set_title(f"Research '{key0}' Confusion Matrix Detail")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                remove_spines(ax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                fig.savefig("figures/research_confusion_matrix_detail.png", dpi=300)
                plt.close(fig)
            else:
                print("No prediction/ground truth data for research key:", key0)
        else:
            print("No research data available for detailed confusion matrix.")
    except Exception as e:
        print("Error creating detailed Research Confusion Matrix plot:", e)
        plt.close()

if __name__ == "__main__":
    main()