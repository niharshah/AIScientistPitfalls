import os, pathlib, random, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------- experiment data dict ----------
experiment_data = {
    "MaskedLanguageModelingPretrain": {
        "SPR_BENCH": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# ---------- working dir ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- device ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- utility --------------
def same_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


same_seed()


# ---------- load SPR -------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    d = DatasetDict()
    d["train"], d["dev"], d["test"] = (
        _load("train.csv"),
        _load("dev.csv"),
        _load("test.csv"),
    )
    return d


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)


# ---------- helper metrics ----------
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def comp_weight(seq):
    return count_shape_variety(seq) * count_color_variety(seq)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [comp_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ---------- vocab & labels ----------
vocab = {"<pad>": 0, "<unk>": 1}
for s in spr["train"]["sequence"]:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
vocab["<mask>"] = len(vocab)  # special mask token
pad_id = vocab["<pad>"]
unk_id = vocab["<unk>"]
mask_id = vocab["<mask>"]
labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}
num_classes = len(labels)
print(f"Vocabulary size = {len(vocab)}, num classes = {num_classes}")


# ---------- datasets --------------
class SPRDatasetMLM(Dataset):
    def __init__(self, sequences, mlm_prob=0.15):
        self.seqs = sequences
        self.mlm_prob = mlm_prob

    def encode(self, toks):
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        ids = self.encode(toks)
        input_ids = ids.copy()
        labels = [-100] * len(ids)
        for i in range(len(ids)):
            if random.random() < self.mlm_prob:
                labels[i] = ids[i]
                input_ids[i] = mask_id
        return input_ids, labels


def collate_mlm(batch):
    ids, lbls = zip(*batch)
    lens = [len(s) for s in ids]
    max_len = max(lens)
    arr = np.full((len(ids), max_len), pad_id, dtype=np.int64)
    lab = np.full((len(ids), max_len), -100, dtype=np.int64)
    for i, (seq, lab_seq) in enumerate(zip(ids, lbls)):
        arr[i, : len(seq)] = seq
        lab[i, : len(seq)] = lab_seq
    return torch.tensor(arr), torch.tensor(lens), torch.tensor(lab)


class SPRDatasetCLS(Dataset):
    def __init__(self, seqs, labels_):
        self.seqs = seqs
        self.labels = [lab2id[l] for l in labels_]

    def encode(self, toks):
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.encode(self.seqs[idx].split()), self.labels[idx], self.seqs[idx]


def collate_cls(batch):
    seqs, ys, raw = zip(*batch)
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    arr = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return (torch.tensor(arr), torch.tensor(lens), torch.tensor(ys)), list(raw)


# ---------- model ------------------
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hid, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid * 2, 128)

    def forward(self, x, lens, *, project=True, return_seq=False):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, h = self.gru(packed)
        if return_seq:
            seq, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, padding_value=0.0
            )
            return seq  # B,L,hidden*2
        h = torch.cat([h[0], h[1]], dim=1)
        return self.proj(h) if project else h


class PretrainMLMModel(nn.Module):
    def __init__(self, vocab_sz):
        super().__init__()
        self.enc = Encoder(vocab_sz)
        self.mlm_head = nn.Linear(256, vocab_sz)  # hid*2 = 256

    def forward(self, x, lens):
        seq_out = self.enc(x, lens, project=False, return_seq=True)  # B,L,256
        logits = self.mlm_head(seq_out)  # B,L,V
        return logits


class Classifier(nn.Module):
    def __init__(self, encoder, nclass):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(128, nclass)

    def forward(self, x, lens):
        z = self.enc(x, lens, project=True)
        return self.head(z)


# ---------- MLM pretrain ----------------
def pretrain_encoder_mlm(epochs=3, batch=256, lr=1e-3):
    model = PretrainMLMModel(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        SPRDatasetMLM(spr["train"]["sequence"]),
        batch_size=batch,
        shuffle=True,
        collate_fn=collate_mlm,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tot_loss = 0
        for x, lens, labels in loader:
            x, lens, labels = x.to(device), lens.to(device), labels.to(device)
            logits = model(x, lens)  # B,L,V
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x.size(0)
        print(
            f"Pretrain epoch {ep}: loss={tot_loss/len(loader.dataset):.4f} "
            f"time={time.time()-t0:.1f}s"
        )
    return model.enc


pretrained_enc = pretrain_encoder_mlm()

# ---------- fine-tune classifier -------------
train_loader = DataLoader(
    SPRDatasetCLS(spr["train"]["sequence"], spr["train"]["label"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_cls,
)
dev_loader = DataLoader(
    SPRDatasetCLS(spr["dev"]["sequence"], spr["dev"]["label"]),
    batch_size=512,
    shuffle=False,
    collate_fn=collate_cls,
)

model = Classifier(pretrained_enc, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val, wait, patience = 1e9, 0, 3
for epoch in range(1, 21):
    # ----- train -----
    model.train()
    tr_loss = 0
    for (x, lens, y), _ in train_loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, lens)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * x.size(0)
    tr_loss /= len(train_loader.dataset)

    # ----- validation -----
    model.eval()
    val_loss = 0
    preds, gt, rawseq = [], [], []
    with torch.no_grad():
        for (x, lens, y), raw in dev_loader:
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            out = model(x, lens)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            gt.extend(y.cpu().tolist())
            rawseq.extend(raw)
    val_loss /= len(dev_loader.dataset)
    cwa = comp_weighted_accuracy(
        rawseq, [id2lab[i] for i in gt], [id2lab[i] for i in preds]
    )

    # log
    experiment_data["MaskedLanguageModelingPretrain"]["SPR_BENCH"]["losses"][
        "train"
    ].append(tr_loss)
    experiment_data["MaskedLanguageModelingPretrain"]["SPR_BENCH"]["losses"][
        "val"
    ].append(val_loss)
    experiment_data["MaskedLanguageModelingPretrain"]["SPR_BENCH"]["metrics"][
        "train"
    ].append(
        None
    )  # no train metric
    experiment_data["MaskedLanguageModelingPretrain"]["SPR_BENCH"]["metrics"][
        "val"
    ].append(cwa)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}  CompWA={cwa:.4f}")

    # early stopping
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        wait = 0
        experiment_data["MaskedLanguageModelingPretrain"]["SPR_BENCH"][
            "predictions"
        ] = preds
        experiment_data["MaskedLanguageModelingPretrain"]["SPR_BENCH"][
            "ground_truth"
        ] = gt
        best_state = model.state_dict()
    else:
        wait += 1
    if wait >= patience:
        break

model.load_state_dict(best_state)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Training finished. Data saved to ./working/experiment_data.npy")
