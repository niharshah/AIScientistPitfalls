import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

fig_count, MAX_FIGS = 0, 5
final_metrics = {}

for dname, dct in experiment_data.items():
    # ---------- 1) LOSS CURVE ----------
    if fig_count < MAX_FIGS:
        try:
            tr_epochs, tr_losses = zip(*dct["losses"]["train"])
            val_epochs, val_losses = zip(*dct["losses"]["val"])
            plt.figure()
            plt.plot(tr_epochs, tr_losses, label="Train")
            plt.plot(val_epochs, val_losses, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname} Loss Curve\nLeft: Train, Right: Val")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"loss_curve_{dname}.png"))
            plt.close()
            fig_count += 1
        except Exception as e:
            print(f"Error plotting loss for {dname}: {e}")
            plt.close()

    # ---------- 2) METRIC CURVES ----------
    if fig_count < MAX_FIGS:
        try:
            epochs, cwa, swa, pcwa = [], [], [], []
            for ep, md in dct["metrics"]["val"]:
                epochs.append(ep)
                cwa.append(md.get("CWA", np.nan))
                swa.append(md.get("SWA", np.nan))
                pcwa.append(md.get("PCWA", np.nan))
            plt.figure()
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, pcwa, label="PCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{dname} Validation Metrics\nCWA, SWA, PCWA vs Epoch")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"metric_curve_{dname}.png"))
            plt.close()
            fig_count += 1
        except Exception as e:
            print(f"Error plotting metrics for {dname}: {e}")
            plt.close()

    # store final metrics for summary print/plot
    try:
        _, last_metrics = dct["metrics"]["val"][-1]
        final_metrics[dname] = last_metrics
    except Exception:
        pass

# ---------- 3) SUMMARY BAR (cross-dataset) ----------
if final_metrics and fig_count < MAX_FIGS:
    try:
        dsets = list(final_metrics.keys())
        pcwa_vals = [final_metrics[d]["PCWA"] for d in dsets]
        x = np.arange(len(dsets))
        plt.figure()
        plt.bar(x, pcwa_vals, width=0.6, color="skyblue")
        plt.xticks(x, dsets, rotation=45, ha="right")
        plt.ylabel("PCWA")
        plt.title("Final-Epoch PCWA Across Datasets\nHigher is Better")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "summary_PCWA_across_datasets.png"))
        plt.close()
        fig_count += 1
    except Exception as e:
        print(f"Error creating summary PCWA plot: {e}")
        plt.close()

# ---------- PRINT FINAL METRICS ----------
if final_metrics:
    print("\nFinal Validation Metrics per Dataset")
    for d, m in final_metrics.items():
        print(
            f"{d}: CWA {m.get('CWA', 'NA'):.4f}, SWA {m.get('SWA', 'NA'):.4f}, PCWA {m.get('PCWA', 'NA'):.4f}"
        )
