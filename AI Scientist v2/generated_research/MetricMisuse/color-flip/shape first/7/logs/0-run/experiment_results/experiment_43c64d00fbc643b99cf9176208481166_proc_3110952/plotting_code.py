import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def get_series(variant_key, metric_key):
    # returns dict hidden_size -> (epochs, values)
    out = {}
    vdict = (
        experiment_data.get(variant_key, {}).get("SPR_BENCH", {}).get("hidden_size", {})
    )
    for hs, run in vdict.items():
        series = run[metric_key]  # list of tuples
        ep, vals = zip(*series) if series else ([], [])
        out[hs] = (list(ep), list(vals))
    return out


# ---------- collect data ----------
loss_train_fs = (
    get_series("aggregation_head_final_state", "losses")["train"] if False else None
)


# quick utility to fetch nested:
def nested_fetch(variant, hs, keypath):
    d = experiment_data[variant]["SPR_BENCH"]["hidden_size"][hs]
    for k in keypath.split("/"):
        d = d[k]
    return d


# build  dictionaries
loss_train_fs, loss_val_fs, hwa_fs = {}, {}, {}
loss_train_mp, loss_val_mp, hwa_mp = {}, {}, {}

for hs in (
    experiment_data.get("aggregation_head_final_state", {})
    .get("SPR_BENCH", {})
    .get("hidden_size", {})
):
    train = experiment_data["aggregation_head_final_state"]["SPR_BENCH"]["hidden_size"][
        hs
    ]["losses"]["train"]
    val = experiment_data["aggregation_head_final_state"]["SPR_BENCH"]["hidden_size"][
        hs
    ]["losses"]["val"]
    met = experiment_data["aggregation_head_final_state"]["SPR_BENCH"]["hidden_size"][
        hs
    ]["metrics"]["val"]
    loss_train_fs[hs] = (list(zip(*train))[0], list(zip(*train))[1])
    loss_val_fs[hs] = (list(zip(*val))[0], list(zip(*val))[1])
    hwa_fs[hs] = (list(zip(*met))[0], list(zip(*met))[3])

for hs in (
    experiment_data.get("aggregation_head_mean_pool", {})
    .get("SPR_BENCH", {})
    .get("hidden_size", {})
):
    train = experiment_data["aggregation_head_mean_pool"]["SPR_BENCH"]["hidden_size"][
        hs
    ]["losses"]["train"]
    val = experiment_data["aggregation_head_mean_pool"]["SPR_BENCH"]["hidden_size"][hs][
        "losses"
    ]["val"]
    met = experiment_data["aggregation_head_mean_pool"]["SPR_BENCH"]["hidden_size"][hs][
        "metrics"
    ]["val"]
    loss_train_mp[hs] = (list(zip(*train))[0], list(zip(*train))[1])
    loss_val_mp[hs] = (list(zip(*val))[0], list(zip(*val))[1])
    hwa_mp[hs] = (list(zip(*met))[0], list(zip(*met))[3])

# ---------- plotting ----------
plots = []

# 1: Final-State loss curves
try:
    plt.figure()
    for hs in sorted(loss_train_fs):
        ep, tr = loss_train_fs[hs]
        _, vl = loss_val_fs[hs]
        plt.plot(ep, tr, label=f"{hs}-train")
        plt.plot(ep, vl, "--", label=f"{hs}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Final-State Head: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_final_state.png")
    plt.savefig(fname)
    plots.append(fname)
except Exception as e:
    print(f"Error creating plot Final-State loss: {e}")
finally:
    plt.close()

# 2: Mean-Pool loss curves
try:
    plt.figure()
    for hs in sorted(loss_train_mp):
        ep, tr = loss_train_mp[hs]
        _, vl = loss_val_mp[hs]
        plt.plot(ep, tr, label=f"{hs}-train")
        plt.plot(ep, vl, "--", label=f"{hs}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Mean-Pool Head: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_mean_pool.png")
    plt.savefig(fname)
    plots.append(fname)
except Exception as e:
    print(f"Error creating plot Mean-Pool loss: {e}")
finally:
    plt.close()

# 3: Final-State HWA curves
try:
    plt.figure()
    for hs in sorted(hwa_fs):
        ep, hv = hwa_fs[hs]
        plt.plot(ep, hv, label=f"{hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH Final-State Head: Validation HWA over Epochs")
    plt.legend(title="Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves_final_state.png")
    plt.savefig(fname)
    plots.append(fname)
except Exception as e:
    print(f"Error creating plot Final-State HWA: {e}")
finally:
    plt.close()

# 4: Mean-Pool HWA curves
try:
    plt.figure()
    for hs in sorted(hwa_mp):
        ep, hv = hwa_mp[hs]
        plt.plot(ep, hv, label=f"{hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH Mean-Pool Head: Validation HWA over Epochs")
    plt.legend(title="Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves_mean_pool.png")
    plt.savefig(fname)
    plots.append(fname)
except Exception as e:
    print(f"Error creating plot Mean-Pool HWA: {e}")
finally:
    plt.close()

# 5: Final epoch HWA comparison
try:
    plt.figure()
    sizes = sorted(set(list(hwa_fs.keys()) + list(hwa_mp.keys())))
    x = np.arange(len(sizes))
    bar_w = 0.35
    fs_final = [hwa_fs[hs][1][-1] for hs in sizes]
    mp_final = [hwa_mp[hs][1][-1] for hs in sizes]
    plt.bar(x - bar_w / 2, fs_final, width=bar_w, label="Final-State")
    plt.bar(x + bar_w / 2, mp_final, width=bar_w, label="Mean-Pool")
    plt.xticks(x, [str(s) for s in sizes])
    plt.xlabel("Hidden Size")
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR_BENCH: Final Epoch HWA vs Hidden Size")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_comparison.png")
    plt.savefig(fname)
    plots.append(fname)
except Exception as e:
    print(f"Error creating plot HWA comparison: {e}")
finally:
    plt.close()

# ---------- print summary ----------
print("Final Epoch HWA")
print("Hidden | Final-State | Mean-Pool")
for hs in sizes:
    print(
        f"{hs:6} | {fs_final[sizes.index(hs)]:10.4f} | {mp_final[sizes.index(hs)]:9.4f}"
    )

print(f"Saved {len(plots)} figures to {working_dir}")
