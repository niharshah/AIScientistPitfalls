import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Helper: fetch dict keyed by embedding size
exp_by_emb = experiment_data.get("embedding_dim", {})

# -------------------- plot 1: loss curves --------------------
try:
    if exp_by_emb:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for emb, entry in exp_by_emb.items():
            axes[0].plot(entry["losses"]["train"], label=f"emb={emb}")
            axes[1].plot(entry["losses"]["val"], label=f"emb={emb}")
        axes[0].set_title("Train Loss")
        axes[1].set_title("Validation Loss")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.suptitle("SPR_BENCH: Loss Curves across Embedding Sizes")
        out_path = os.path.join(working_dir, "SPR_BENCH_loss_curves_by_emb.png")
        plt.savefig(out_path)
        print(f"Saved {out_path}")
        plt.close(fig)
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# -------------------- plot 2: validation HWA per epoch --------------------
try:
    if exp_by_emb:
        plt.figure(figsize=(6, 4))
        for emb, entry in exp_by_emb.items():
            hwa = [m[2] for m in entry["metrics"]["val"]]
            plt.plot(range(1, len(hwa) + 1), hwa, marker="o", label=f"emb={emb}")
        plt.title("SPR_BENCH: Validation HWA vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        out_path = os.path.join(working_dir, "SPR_BENCH_val_HWA.png")
        plt.savefig(out_path)
        print(f"Saved {out_path}")
        plt.close()
except Exception as e:
    print(f"Error creating validation HWA figure: {e}")
    plt.close()

# -------------------- plot 3: test HWA bar chart --------------------
try:
    if exp_by_emb:
        embeds, hwa_vals = [], []
        for emb, entry in exp_by_emb.items():
            embeds.append(str(emb))
            hwa_vals.append(entry["metrics"]["test"][2])
        plt.figure(figsize=(6, 4))
        plt.bar(embeds, hwa_vals, color="skyblue")
        plt.title("SPR_BENCH: Test HWA by Embedding Size")
        plt.xlabel("Embedding Dim")
        plt.ylabel("HWA")
        out_path = os.path.join(working_dir, "SPR_BENCH_test_HWA.png")
        plt.savefig(out_path)
        print(f"Saved {out_path}")
        plt.close()
except Exception as e:
    print(f"Error creating test HWA figure: {e}")
    plt.close()

# -------------------- print test metrics --------------------
for emb, entry in exp_by_emb.items():
    swa, cwa, hwa = entry["metrics"]["test"]
    print(f"emb={emb:>3}:  Test SWA={swa:.4f}  CWA={cwa:.4f}  HWA={hwa:.4f}")
