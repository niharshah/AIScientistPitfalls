import os, pathlib, random, numpy as np, torch, copy, warnings
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict

warnings.filterwarnings("ignore")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- metrics ----------
def count_color(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape(seq):
    return len({tok[0] for tok in seq.split() if tok})


def CWA(seqs, y_t, y_p):
    w = [count_color(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def SWA(seqs, y_t, y_p):
    w = [count_shape(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def CompWA(seqs, y_t, y_p):
    w = [count_color(s) * count_shape(s) for s in seqs]
    c = [w_ if t == p else 0 for w_, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


# ---------- data ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def get_dataset() -> DatasetDict:
    spr_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if spr_path.exists():
        return load_spr_bench(spr_path)
    shapes, colors = ["▲", "■", "●", "◆"], list("RGBY")

    def gen(n):
        seqs, labels, ids = [], [], []
        for i in range(n):
            toks = [
                random.choice(shapes) + random.choice(colors)
                for _ in range(random.randint(4, 10))
            ]
            seqs.append(" ".join(toks))
            labels.append(random.choice(["ruleA", "ruleB", "ruleC"]))
            ids.append(str(i))
        return Dataset.from_dict({"id": ids, "sequence": seqs, "label": labels})

    return DatasetDict(train=gen(800), dev=gen(200), test=gen(200))


spr = get_dataset()


# ---------- vocab ----------
def make_stoi(split):
    vocab = set()
    [vocab.update(s.split()) for s in split["sequence"]]
    return {tok: i + 1 for i, tok in enumerate(sorted(vocab))}  # 0 -> PAD


# ---------- torch dataset ----------
class SPRTorch(torch.utils.data.Dataset):
    def __init__(self, hf_split, stoi, label2id):
        self.raw = hf_split["sequence"]
        self.lbl = [label2id[l] for l in hf_split["label"]]
        self.stoi = stoi

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        ids = [self.stoi[t] for t in self.raw[idx].split()]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(self.lbl[idx]),
            "raw": self.raw[idx],
        }


def collate(batch):
    mx = max(len(x["input_ids"]) for x in batch)
    inp = torch.stack(
        [
            nn.functional.pad(x["input_ids"], (0, mx - len(x["input_ids"])), value=0)
            for x in batch
        ]
    )
    lbl = torch.stack([x["labels"] for x in batch])
    raw = [x["raw"] for x in batch]
    return {"input_ids": inp, "labels": lbl, "raw": raw}


# ---------- model ----------
class EncoderClassifier(nn.Module):
    def __init__(self, vocab, emb=32, hidden=64, classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, classes)

    def forward(self, x):
        e = self.embedding(x)
        _, h = self.rnn(e)
        h = torch.cat([h[0], h[1]], 1)
        return self.fc(h)


experiment_data = {}


# ---------- experiment runner ----------
def run_experiment(ablation_name, cluster_mode="mean", seed=0, lr=2e-3, epochs=4):
    global experiment_data
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    stoi = make_stoi(spr["train"])
    label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
    train_dl = lambda s: DataLoader(
        SPRTorch(spr["train"], s, label2id),
        batch_size=64,
        shuffle=True,
        collate_fn=collate,
    )
    dev_dl = lambda s: DataLoader(
        SPRTorch(spr["dev"], s, label2id),
        batch_size=128,
        shuffle=False,
        collate_fn=collate,
    )
    tr_dl = train_dl(stoi)
    dv_dl = dev_dl(stoi)
    model = EncoderClassifier(len(stoi) + 1, classes=len(label2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    experiment_data[ablation_name] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }

    clustered = False
    for epoch in range(1, epochs + 1):
        # -- train
        model.train()
        tot = 0
        for b in tr_dl:
            bt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            opt.zero_grad()
            out = model(bt["input_ids"])
            loss = loss_fn(out, bt["labels"])
            loss.backward()
            opt.step()
            tot += loss.item() * bt["labels"].size(0)
        tr_loss = tot / len(tr_dl.dataset)
        experiment_data[ablation_name]["SPR_BENCH"]["losses"]["train"].append(tr_loss)

        # -- eval
        model.eval()
        vloss = 0
        preds = []
        gts = []
        raws = []
        with torch.no_grad():
            for b in dv_dl:
                bt = {
                    k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()
                }
                out = model(bt["input_ids"])
                loss = loss_fn(out, bt["labels"])
                vloss += loss.item() * bt["labels"].size(0)
                p = torch.argmax(out, 1).cpu().tolist()
                preds.extend(p)
                gts.extend(bt["labels"].cpu().tolist())
                raws.extend(b["raw"])
        vloss /= len(dv_dl.dataset)
        acc = float(np.mean([p == g for p, g in zip(preds, gts)]))
        cwa, swa, comp = (
            float(CWA(raws, gts, preds)),
            float(SWA(raws, gts, preds)),
            float(CompWA(raws, gts, preds)),
        )
        experiment_data[ablation_name]["SPR_BENCH"]["losses"]["val"].append(vloss)
        experiment_data[ablation_name]["SPR_BENCH"]["metrics"]["val"].append(
            {"epoch": epoch, "acc": acc, "CWA": cwa, "SWA": swa, "CompWA": comp}
        )
        print(f"[{ablation_name}] Epoch {epoch}: val_loss={vloss:.4f} ACC={acc:.3f}")

        # -- random clustering after first epoch
        if not clustered:
            with torch.no_grad():
                emb_np = model.embedding.weight[1:].detach().cpu().numpy()
            k = min(16, emb_np.shape[0])
            rng = np.random.RandomState(seed)
            rand_labels = rng.randint(0, k, size=emb_np.shape[0])
            token_to_cluster = {
                tok: cid + 1
                for tok, cid in zip(sorted(stoi, key=lambda x: stoi[x]), rand_labels)
            }
            new_emb = nn.Embedding(
                k + 1, model.embedding.embedding_dim, padding_idx=0
            ).to(device)
            if cluster_mode == "mean":
                for cid in range(k):
                    idxs = np.where(rand_labels == cid)[0]
                    if len(idxs):
                        mean_vec = emb_np[idxs].mean(0)
                        new_emb.weight.data[cid + 1] = torch.tensor(
                            mean_vec, device=device
                        )
            model.embedding = new_emb
            stoi = {tok: token_to_cluster[tok] for tok in stoi}
            tr_dl = train_dl(stoi)
            dv_dl = dev_dl(stoi)
            clustered = True
            print(
                f"[{ablation_name}] Random clustering done. k={k}, mode={cluster_mode}"
            )
    experiment_data[ablation_name]["SPR_BENCH"]["predictions"] = preds
    experiment_data[ablation_name]["SPR_BENCH"]["ground_truth"] = gts


# ---------- run both variants ----------
run_experiment("random_cluster_mean", cluster_mode="mean", seed=42)
run_experiment("random_cluster_rand", cluster_mode="rand", seed=42)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
