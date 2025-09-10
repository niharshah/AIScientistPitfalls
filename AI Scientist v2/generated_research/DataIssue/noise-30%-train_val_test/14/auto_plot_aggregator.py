"""
Final Aggregator Script for SPR_BENCH Final Figures

This script loads experiment results (stored in .npy files) generated for the Baseline, Research, and Ablation experiments.
It aggregates the key numerical metrics and produces a set of final publication‐quality figures,
which are stored solely in the "figures/" directory.

Notes:
– All .npy file paths are hard‐coded exactly as provided in the experiment summaries.
– Each figure is produced in its own try/except block so that one failure does not break others.
– Figures are saved with high dpi and with professional style (e.g. increased font sizes, no top/right spines).
– Aggregated figures include separate comparisons for Baseline, Research, and Ablation experiments.
– Ablation experiments are compared both individually (final metric bar charts) and across epochs.

Before running, please ensure that the necessary .npy files are available at their full paths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set global styles for publication-quality figures
plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
DPI = 300

# Create output directory for figures
os.makedirs("figures", exist_ok=True)

# ---------------- Helper Functions ----------------

def load_experiment_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def save_figure(fig, fname):
    try:
        fig.savefig(os.path.join("figures", fname), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error saving figure {fname}: {e}")
        plt.close(fig)

def extract_baseline_metrics(baseline_data):
    """
    For the baseline experiment the structure is assumed to be:
      { "d_model_tuning": { dm: { "SPR_BENCH": { 
                      "losses": { "train": [ { "epoch": int, "loss": float }, ... ],
                                   "val": [ { "epoch": int, "loss": float }, ... ] },
                      "metrics": { "train": [ { "epoch": int, "macro_f1": float }, ... ],
                                   "val": [ { "epoch": int, "macro_f1": float }, ... ] },
                      ... } } } }
    Returns: dictionary mapping each d_model to its train and val series and final val macro-f1.
    """
    results = {}
    dct = baseline_data.get("d_model_tuning", {})
    for dm in sorted(dct.keys(), key=int):
        run = dct[dm].get("SPR_BENCH", {})
        epochs = [rec["epoch"] for rec in run.get("metrics", {}).get("train", [])]
        loss_train = [rec["loss"] for rec in run.get("losses", {}).get("train", [])]
        loss_val = [rec["loss"] for rec in run.get("losses", {}).get("val", [])]
        f1_train = [rec["macro_f1"] for rec in run.get("metrics", {}).get("train", [])]
        f1_val = [rec["macro_f1"] for rec in run.get("metrics", {}).get("val", [])]
        final_f1 = f1_val[-1] if f1_val else 0.0
        results[dm] = {
            "epochs": epochs,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "f1_train": f1_train,
            "f1_val": f1_val,
            "final_f1": final_f1
        }
    return results

def extract_research_metrics(research_data):
    """
    For research summary file, data is keyed by model names under "SPR_BENCH".
    Returns a dict with keys as model names and values are dictionaries of series.
    """
    results = {}
    runs = research_data.get("SPR_BENCH", {})
    for model in runs:
        rec = runs[model]
        epochs = [item["epoch"] for item in rec.get("metrics", {}).get("train", [])]
        loss_train = [item["loss"] for item in rec.get("losses", {}).get("train", [])]
        loss_val = [item["loss"] for item in rec.get("losses", {}).get("val", [])]
        f1_train = [item["macro_f1"] for item in rec.get("metrics", {}).get("train", [])]
        f1_val = [item["macro_f1"] for item in rec.get("metrics", {}).get("val", [])]
        final_f1 = f1_val[-1] if f1_val else 0.0
        results[model] = {
            "epochs": epochs,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "f1_train": f1_train,
            "f1_val": f1_val,
            "final_f1": final_f1,
            "predictions": rec.get("predictions", []),
            "ground_truth": rec.get("ground_truth", [])
        }
    return results

def extract_ablation_final_metrics(ablation_data, key_name):
    """
    Ablation experiments are assumed to be stored under key "SPR_BENCH" with model entries.
    key_name: the model key to extract from the ablation experiment.
    Returns final validation macro_f1 and final RGA (accuracy) from that model.
    """
    spr = ablation_data.get("SPR_BENCH", {})
    if key_name not in spr:
        return None, None, None  # Not found
    rec = spr[key_name]
    f1_series = rec.get("metrics", {}).get("val", [])
    final_macro_f1 = f1_series[-1]["macro_f1"] if f1_series else 0.0
    # Some ablation logs include "RGA" under metrics:
    acc_series = rec.get("metrics", {}).get("val", [])
    final_RGA = acc_series[-1].get("RGA", 0.0) if acc_series else 0.0
    # Also attempt to extract full f1 curve (epochs and val curve)
    epochs = [rec_item["epoch"] for rec_item in rec.get("metrics", {}).get("train", [])]
    f1_val_curve = [rec_item["macro_f1"] for rec_item in rec.get("metrics", {}).get("val", [])]
    return final_macro_f1, final_RGA, (epochs, f1_val_curve)

def plot_line(ax, x, y, label, linestyle="-"):
    ax.plot(x, y, linestyle=linestyle, marker="o", label=label)

def add_legend(ax):
    ax.legend()

# ---------------- Load Experiment Data from Provided File Paths ----------------

# Baseline summary npy file:
baseline_file = "experiment_results/experiment_fd25c9ee7e7e49f89867a83b3163a8b2_proc_3462556/experiment_data.npy"
baseline_data = load_experiment_data(baseline_file)
baseline_metrics = extract_baseline_metrics(baseline_data) if baseline_data else {}

# Research summary npy file:
research_file = "experiment_results/experiment_2b7eac33c2d341b4867b07e86ce4f259_proc_3469037/experiment_data.npy"
research_data = load_experiment_data(research_file)
research_metrics = extract_research_metrics(research_data) if research_data else {}

# Ablation experiments mapping: (model key, file_path)
ablation_experiments = [
    ("FrozenEmb", "experiment_results/experiment_b3f6d90bed2d42dc9d0e5d55fcecd78e_proc_3477820/experiment_data.npy"),
    ("NoSymToken", "experiment_results/experiment_9914cf81164c4847bad35fee7bbb0090_proc_3477821/experiment_data.npy"),
    ("MeanPool", "experiment_results/experiment_c3aca909722e47e58424c60177c37cff_proc_3477819/experiment_data.npy"),
    ("SymToken_Sinusoidal", "experiment_results/experiment_613bee912f6a4d2ebad49c504f786599_proc_3477821/experiment_data.npy"),
    ("RandomSymToken", "experiment_results/experiment_fe6ac296b6614561b977e860df42420b_proc_3477818/experiment_data.npy"),
    ("TiedEmbeddingHead", "experiment_results/experiment_b94241333d1d4f808b5db9d23ba106e0_proc_3477821/experiment_data.npy"),
    ("RelativePosBias", "experiment_results/experiment_ea16362d8b9c49e38bcafb5486200eb5_proc_3477819/experiment_data.npy")
]

ablation_results = {}
for model_key, file_path in ablation_experiments:
    data = load_experiment_data(file_path)
    if data:
        final_f1, final_RGA, curve = extract_ablation_final_metrics(data, model_key)
        if final_f1 is not None:
            ablation_results[model_key] = {
                "final_f1": final_f1,
                "final_RGA": final_RGA,
                "curve": curve  # (epochs, f1_curve)
            }

# ---------------- Create Final Figures ----------------

# Figure 1: Baseline Loss Curves vs Epoch (multiple d_model values)
try:
    fig, ax = plt.subplots(figsize=(8,6))
    for dm, metrics in baseline_metrics.items():
        plot_line(ax, metrics["epochs"], metrics["loss_train"], label=f"{dm} train", linestyle="--")
        plot_line(ax, metrics["epochs"], metrics["loss_val"], label=f"{dm} val", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Baseline: Loss vs Epoch (d_model tuning)")
    add_legend(ax)
    save_figure(fig, "Baseline_Loss_Curves.png")
except Exception as e:
    print(f"Error creating Baseline Loss Curves: {e}")

# Figure 2: Baseline Macro-F1 Curves vs Epoch
try:
    fig, ax = plt.subplots(figsize=(8,6))
    for dm, metrics in baseline_metrics.items():
        plot_line(ax, metrics["epochs"], metrics["f1_train"], label=f"{dm} train", linestyle="--")
        plot_line(ax, metrics["epochs"], metrics["f1_val"], label=f"{dm} val", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title("Baseline: Macro F1 vs Epoch (d_model tuning)")
    add_legend(ax)
    save_figure(fig, "Baseline_MacroF1_Curves.png")
except Exception as e:
    print(f"Error creating Baseline Macro-F1 Curves: {e}")

# Figure 3: Baseline Final Validation Macro-F1 Bar Chart (per d_model)
try:
    fig, ax = plt.subplots(figsize=(8,6))
    dms = list(baseline_metrics.keys())
    final_f1s = [baseline_metrics[dm]["final_f1"] for dm in dms]
    ax.bar(dms, final_f1s, color="mediumseagreen")
    ax.set_xlabel("d_model")
    ax.set_ylabel("Final Validation Macro F1")
    ax.set_title("Baseline: Final Validation Macro F1 per d_model")
    save_figure(fig, "Baseline_Final_Val_MacroF1_Bar.png")
except Exception as e:
    print(f"Error creating Baseline Final Macro F1 Bar: {e}")

# Figure 4: Research Loss Curves vs Epoch (comparing models)
try:
    fig, ax = plt.subplots(figsize=(10,6))
    for model, metrics in research_metrics.items():
        plot_line(ax, metrics["epochs"], metrics["loss_train"], label=f"{model} train", linestyle="--")
        plot_line(ax, metrics["epochs"], metrics["loss_val"], label=f"{model} val", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Research: Loss vs Epoch (Baseline vs SymbolicToken)")
    add_legend(ax)
    save_figure(fig, "Research_Loss_Curves.png")
except Exception as e:
    print(f"Error creating Research Loss Curves: {e}")

# Figure 5: Research Macro-F1 Curves vs Epoch
try:
    fig, ax = plt.subplots(figsize=(10,6))
    for model, metrics in research_metrics.items():
        plot_line(ax, metrics["epochs"], metrics["f1_train"], label=f"{model} train", linestyle="--")
        plot_line(ax, metrics["epochs"], metrics["f1_val"], label=f"{model} val", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title("Research: Macro F1 vs Epoch (Baseline vs SymbolicToken)")
    add_legend(ax)
    save_figure(fig, "Research_MacroF1_Curves.png")
except Exception as e:
    print(f"Error creating Research Macro-F1 Curves: {e}")

# Figure 6: Research Final Validation Macro-F1 Bar Chart
try:
    fig, ax = plt.subplots(figsize=(8,6))
    models_r = list(research_metrics.keys())
    final_f1s_r = [research_metrics[m]["final_f1"] for m in models_r]
    ax.bar(models_r, final_f1s_r, color="slateblue")
    ax.set_xlabel("Model")
    ax.set_ylabel("Final Validation Macro F1")
    ax.set_title("Research: Final Validation Macro F1 per Model")
    save_figure(fig, "Research_Final_Val_MacroF1_Bar.png")
except Exception as e:
    print(f"Error creating Research Final MacroF1 Bar: {e}")

# Figure 7: Research Confusion Matrix for Best Model
try:
    # Select best model (highest final validation macro F1)
    best_model = max(research_metrics.keys(), key=lambda m: research_metrics[m]["final_f1"])
    preds = np.array(research_metrics[best_model].get("predictions"))
    gts = np.array(research_metrics[best_model].get("ground_truth"))
    # Create confusion matrix
    labels = np.unique(np.concatenate((gts, preds)))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Research: Confusion Matrix - {best_model}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    save_figure(fig, f"Research_Confusion_Matrix_{best_model}.png")
except Exception as e:
    print(f"Error creating Research Confusion Matrix: {e}")

# Figure 8: Aggregated Ablation Final Macro-F1 Bar Chart
try:
    fig, ax = plt.subplots(figsize=(10,6))
    keys = list(ablation_results.keys())
    final_f1s_ablation = [ablation_results[k]["final_f1"] for k in keys]
    ax.bar(keys, final_f1s_ablation, color="indianred")
    ax.set_xlabel("Ablation Model")
    ax.set_ylabel("Final Validation Macro F1")
    ax.set_title("Ablation: Final Validation Macro F1 Comparison")
    save_figure(fig, "Ablation_Final_MacroF1_Bar.png")
except Exception as e:
    print(f"Error creating Ablation Final Macro F1 Bar Chart: {e}")

# Figure 9: Ablation Validation Macro-F1 Curve Comparison (plot curves from each ablation)
try:
    fig, ax = plt.subplots(figsize=(10,6))
    for model_key, res in ablation_results.items():
        curve = res.get("curve")
        if curve is not None:
            epochs, f1_curve = curve
            ax.plot(epochs, f1_curve, marker="o", label=model_key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Ablation: Validation Macro F1 Curves Comparison")
    add_legend(ax)
    save_figure(fig, "Ablation_MacroF1_Curves.png")
except Exception as e:
    print(f"Error creating Ablation Macro F1 Curves: {e}")

# Figure 10: Aggregated Ablation Final RGA (Accuracy) Bar Chart
try:
    fig, ax = plt.subplots(figsize=(10,6))
    keys = list(ablation_results.keys())
    final_RGAs = [ablation_results[k]["final_RGA"] for k in keys]
    ax.bar(keys, final_RGAs, color="mediumorchid")
    ax.set_xlabel("Ablation Model")
    ax.set_ylabel("Final Validation Accuracy (RGA)")
    ax.set_title("Ablation: Final Validation Accuracy Comparison")
    save_figure(fig, "Ablation_Final_RGA_Bar.png")
except Exception as e:
    print(f"Error creating Ablation Final RGA Bar Chart: {e}")

# Figure 11: Overall Comparison: Final Macro-F1 Across Baseline, Research Best, and Ablations
try:
    fig, ax = plt.subplots(figsize=(12,6))
    overall_labels = []
    overall_scores = []
    # Add Baseline best d_model from baseline experiments
    best_dm = max(baseline_metrics.keys(), key=lambda dm: baseline_metrics[dm]["final_f1"])
    overall_labels.append(f"Baseline (d_model {best_dm})")
    overall_scores.append(baseline_metrics[best_dm]["final_f1"])
    # Add Research best model
    overall_labels.append(f"Research ({best_model})")
    overall_scores.append(research_metrics[best_model]["final_f1"])
    # Add each ablation
    for k, res in ablation_results.items():
        overall_labels.append(f"Ablation ({k})")
        overall_scores.append(res["final_f1"])
    ax.barh(overall_labels, overall_scores, color="teal")
    ax.set_xlabel("Final Validation Macro F1")
    ax.set_title("Overall Final Macro F1 Comparison")
    for index, value in enumerate(overall_scores):
        ax.text(value, index, f" {value:.2f}", va="center")
    save_figure(fig, "Overall_Final_MacroF1_Comparison.png")
except Exception as e:
    print(f"Error creating Overall Final Macro F1 Comparison: {e}")

print("Final figures have been saved in the 'figures/' directory.")