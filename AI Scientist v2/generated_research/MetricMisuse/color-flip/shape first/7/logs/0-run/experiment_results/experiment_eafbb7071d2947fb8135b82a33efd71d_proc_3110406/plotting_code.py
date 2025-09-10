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


def get_store(hs):
    return experiment_data["unidirectional_LSTM"]["SPR_BENCH"]["hidden_size"][hs]


hidden_sizes = sorted(
    experiment_data.get("unidirectional_LSTM", {})
    .get("SPR_BENCH", {})
    .get("hidden_size", {})
    .keys()
)

# -------- Figure 1 : Loss curves --------
try:
    plt.figure(figsize=(6, 4))
    for hs in hidden_sizes:
        store = get_store(hs)
        epochs = [e for e, _ in store["losses"]["train"]]
        tr_loss = [l for _, l in store["losses"]["train"]]
        va_loss = [l for _, l in store["losses"]["val"]]
        plt.plot(epochs, tr_loss, "--", label=f"train_h{hs}")
        plt.plot(epochs, va_loss, "-", label=f"val_h{hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title("SPR_BENCH UniLSTM Training vs Validation Loss")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "SPR_BENCH_loss_curves_unidirectional_LSTM.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# -------- Figure 2 : HWA curves --------
try:
    plt.figure(figsize=(6, 4))
    for hs in hidden_sizes:
        store = get_store(hs)
        epochs = [e for e, *_ in store["metrics"]["val"]]
        hwa = [h for _, _, _, h in store["metrics"]["val"]]
        plt.plot(epochs, hwa, label=f"h{hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH UniLSTM Validation HWA Across Epochs")
    plt.legend(title="Hidden Size", fontsize=6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "SPR_BENCH_HWA_curves_unidirectional_LSTM.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve figure: {e}")
    plt.close()

# -------- Figure 3 : Final metrics bar chart --------
try:
    swa_fin, cwa_fin, hwa_fin = [], [], []
    for hs in hidden_sizes:
        store = get_store(hs)
        swa_fin.append(store["metrics"]["val"][-1][1])
        cwa_fin.append(store["metrics"]["val"][-1][2])
        hwa_fin.append(store["metrics"]["val"][-1][3])

    x = np.arange(len(hidden_sizes))
    width = 0.25
    plt.figure(figsize=(7, 4))
    plt.bar(x - width, swa_fin, width, label="SWA")
    plt.bar(x, cwa_fin, width, label="CWA")
    plt.bar(x + width, hwa_fin, width, label="HWA")
    plt.xticks(x, [f"h{hs}" for hs in hidden_sizes])
    plt.ylabel("Score")
    plt.title("SPR_BENCH UniLSTM Final Validation Metrics by Hidden Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "SPR_BENCH_final_metrics_unidirectional_LSTM.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating final metrics figure: {e}")
    plt.close()
