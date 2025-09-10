import os, pathlib, random, numpy as np, torch, copy, warnings
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.cluster import KMeans

# ---------- misc ----------
warnings.filterwarnings("ignore", category=UserWarning)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- experiment store ----------
experiment_data = {}


# ---------- data (SPR_BENCH OR toy) ----------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _l(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_l("train.csv"), dev=_l("dev.csv"), test=_l("test.csv"))


def get_dataset() -> DatasetDict:
    spr_path = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
    if spr_path.exists():
        return load_spr_bench(spr_path)
    # toy fallback
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


# ---------- metrics ----------
def count_color(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape(seq):
    return len({tok[0] for tok in seq.split() if tok})


def _W(seqs, f):
    return [f(s) for s in seqs]


def _score(seqs, y_t, y_p, w_func):
    w = _W(seqs, w_func)
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) > 0 else 0.0


def CWA(s, y_t, y_p):
    return _score(s, y_t, y_p, count_color)


def SWA(s, y_t, y_p):
    return _score(s, y_t, y_p, count_shape)


def CompWA(s, y_t, y_p):
    return _score(s, y_t, y_p, lambda s: count_color(s) * count_shape(s))


# ---------- vocab ----------
def make_stoi(split):
    vocab = set()
    [vocab.update(s.split()) for s in split["sequence"]]
    return {tok: i + 1 for i, tok in enumerate(sorted(vocab))}  # 0=PAD


# ---------- torch dataset ----------
class SPRTorch(torch.utils.data.Dataset):
    def __init__(self, hf_split, stoi, label2id):
        self.raw = hf_split["sequence"]
        self.ids = [[stoi[t] for t in seq.split()] for seq in self.raw]
        self.lbl = [label2id[l] for l in hf_split["label"]]

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.ids[idx]),
            "labels": torch.tensor(self.lbl[idx]),
            "raw": self.raw[idx],
        }


def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.stack(
        [
            nn.functional.pad(b["input_ids"], (0, m - len(b["input_ids"])), value=0)
            for b in batch
        ]
    )
    lbl = torch.stack([b["labels"] for b in batch])
    raw = [b["raw"] for b in batch]
    return {"input_ids": ids, "labels": lbl, "raw": raw}


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


# ---------- training ----------
def train_loop(exp_name: str, reinit_rnn: bool = False, lr=2e-3, epochs=4):
    print(f"\n=== Running experiment: {exp_name} | reinit_rnn={reinit_rnn} ===")
    experiment_data[exp_name] = {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    stoi = make_stoi(spr["train"])
    label2id = {l: i for i, l in enumerate(sorted(set(spr["train"]["label"])))}
    tr_dl = lambda: DataLoader(
        SPRTorch(spr["train"], stoi, label2id),
        batch_size=64,
        shuffle=True,
        collate_fn=collate,
    )
    dv_dl = lambda: DataLoader(
        SPRTorch(spr["dev"], stoi, label2id),
        batch_size=128,
        shuffle=False,
        collate_fn=collate,
    )
    train_dl, dev_dl = tr_dl(), dv_dl()
    model = EncoderClassifier(len(stoi) + 1, classes=len(label2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    clustered = False
    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot = 0
        for b in train_dl:
            b_t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            opt.zero_grad()
            out = model(b_t["input_ids"])
            loss = loss_fn(out, b_t["labels"])
            loss.backward()
            opt.step()
            tot += loss.item() * b_t["labels"].size(0)
        tr_loss = tot / len(train_dl.dataset)
        experiment_data[exp_name]["SPR_BENCH"]["losses"]["train"].append(tr_loss)
        # ---- validate ----
        model.eval()
        vtot = 0
        preds = []
        gts = []
        raws = []
        with torch.no_grad():
            for b in dev_dl:
                b_t = {
                    k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()
                }
                out = model(b_t["input_ids"])
                loss = loss_fn(out, b_t["labels"])
                vtot += loss.item() * b_t["labels"].size(0)
                p = torch.argmax(out, 1).cpu().tolist()
                preds.extend(p)
                gts.extend(b_t["labels"].cpu().tolist())
                raws.extend(b["raw"])
        v_loss = vtot / len(dev_dl.dataset)
        acc = float(np.mean([p == g for p, g in zip(preds, gts)]))
        cwa, swa, comp = (
            float(CWA(raws, gts, preds)),
            float(SWA(raws, gts, preds)),
            float(CompWA(raws, gts, preds)),
        )
        experiment_data[exp_name]["SPR_BENCH"]["losses"]["val"].append(v_loss)
        experiment_data[exp_name]["SPR_BENCH"]["metrics"]["val"].append(
            {"epoch": ep, "acc": acc, "CWA": cwa, "SWA": swa, "CompWA": comp}
        )
        print(
            f"Epoch {ep}: val_loss={v_loss:.4f} | ACC={acc:.3f} CWA={cwa:.3f} SWA={swa:.3f} CompWA={comp:.3f}"
        )
        # ---- latent glyph clustering after first epoch ----
        if not clustered:
            print("Running latent glyph clustering …")
            with torch.no_grad():
                emb_np = model.embedding.weight[1:].detach().cpu().numpy()
            k = min(16, emb_np.shape[0])
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb_np)
            token_to_cluster = {
                tok: cid + 1
                for tok, cid in zip(sorted(stoi, key=lambda x: stoi[x]), km.labels_)
            }
            new_emb = nn.Embedding(
                k + 1, model.embedding.embedding_dim, padding_idx=0
            ).to(device)
            new_emb.weight.data[1:] = torch.tensor(km.cluster_centers_, device=device)
            model.embedding = new_emb
            stoi = {tok: token_to_cluster[tok] for tok in stoi}
            train_dl, dev_dl = tr_dl(), dv_dl()
            # --- optional RNN reinitialisation ---
            if reinit_rnn:
                hidden = model.rnn.hidden_size
                inp_dim = model.embedding.embedding_dim
                bid = model.rnn.bidirectional
                classes = model.fc.out_features
                model.rnn = nn.GRU(
                    inp_dim, hidden, batch_first=True, bidirectional=bid
                ).to(device)
                model.fc = nn.Linear(hidden * 2 if bid else hidden, classes).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                print("GRU and classifier re-initialised.")
            clustered = True
            print(f"Clustering done. New vocab size={k}")
    experiment_data[exp_name]["SPR_BENCH"]["predictions"] = preds
    experiment_data[exp_name]["SPR_BENCH"]["ground_truth"] = gts


# ---------- run both baseline and ablation ----------
train_loop("baseline_cluster", reinit_rnn=False)
train_loop("reinit_rnn_after_clustering", reinit_rnn=True)

# ---------- save ----------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
