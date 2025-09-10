import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ---------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ---------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    wd_data = experiment_data.get("weight_decay", {})
    wds = sorted(wd_data.keys(), key=float)  # sort numerically as strings
    epochs = len(next(iter(wd_data.values()))["SPR_BENCH"]["losses"]["train"])

    # ----- gather tensors -----
    tr_loss, va_loss, cwa, swa, gcwa, test_m = {}, {}, {}, {}, {}, {}
    for wd in wds:
        d = wd_data[wd]["SPR_BENCH"]
        tr_loss[wd] = d["losses"]["train"]
        va_loss[wd] = d["losses"]["val"]
        cwa[wd] = [m["CWA"] for m in d["metrics"]["val"]]
        swa[wd] = [m["SWA"] for m in d["metrics"]["val"]]
        gcwa[wd] = [m["GCWA"] for m in d["metrics"]["val"]]
        test_m[wd] = d["metrics"]["test"]

    # Common style helpers
    epoch_idx = list(range(1, epochs + 1))
    colours = plt.cm.tab10(np.linspace(0, 1, len(wds)))

    # ---------- 1. Loss curves ----------
    try:
        plt.figure()
        for i, wd in enumerate(wds):
            plt.plot(
                epoch_idx,
                tr_loss[wd],
                linestyle="--",
                color=colours[i],
                label=f"wd={wd} train",
            )
            plt.plot(
                epoch_idx,
                va_loss[wd],
                linestyle="-",
                color=colours[i],
                label=f"wd={wd} val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 2. CWA curves ----------
    try:
        plt.figure()
        for i, wd in enumerate(wds):
            plt.plot(epoch_idx, cwa[wd], color=colours[i], label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.title("SPR_BENCH: Color-Weighted Accuracy across Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_CWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # ---------- 3. SWA curves ----------
    try:
        plt.figure()
        for i, wd in enumerate(wds):
            plt.plot(epoch_idx, swa[wd], color=colours[i], label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH: Shape-Weighted Accuracy across Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_SWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ---------- 4. GCWA curves ----------
    try:
        plt.figure()
        for i, wd in enumerate(wds):
            plt.plot(epoch_idx, gcwa[wd], color=colours[i], label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("GCWA")
        plt.title("SPR_BENCH: Glyph-Complexity-Weighted Accuracy across Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_GCWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating GCWA plot: {e}")
        plt.close()

    # ---------- 5. Test metric bar chart ----------
    try:
        labels = ["CWA", "SWA", "GCWA"]
        x = np.arange(len(labels))
        width = 0.18
        plt.figure()
        for i, wd in enumerate(wds):
            vals = [test_m[wd][k] for k in labels]
            plt.bar(x + i * width, vals, width, label=f"wd={wd}", color=colours[i])
        plt.xticks(x + width * (len(wds) - 1) / 2, labels)
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Test Metrics by Weight Decay")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # ---------- print table ----------
    print("\n=== Test Metrics (SPR_BENCH) ===")
    for wd in wds:
        t = test_m[wd]
        print(
            f"wd={wd:>6}: CWA={t['CWA']:.3f}, SWA={t['SWA']:.3f}, GCWA={t['GCWA']:.3f}"
        )
