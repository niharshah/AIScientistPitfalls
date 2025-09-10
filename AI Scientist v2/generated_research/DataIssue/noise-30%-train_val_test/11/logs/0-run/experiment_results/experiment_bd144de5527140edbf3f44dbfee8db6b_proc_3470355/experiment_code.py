import os, math, random, time, pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# --- working dir ----------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- experiment data holder -----------------------------------------------------------
experiment_data = {
    "SPR_BENCH_MLM": {
        "metrics": {"train_macro_f1": [], "val_macro_f1": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}


# --- reproducibility ------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# --- device ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- load SPR_BENCH -------------------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(root / split_csv),
            split="train",
            cache_dir=".cache_dsets",
        )

    dd = DatasetDict()
    for sp, fn in zip(["train", "dev", "test"], ["train.csv", "dev.csv", "test.csv"]):
        dd[sp] = _load(fn)
    return dd


data_root = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(data_root)
print({k: len(v) for k, v in spr.items()})

# --- vocabulary -----------------------------------------------------------------------
PAD, UNK, CLS, MSK = "<pad>", "<unk>", "<cls>", "<mask>"


def build_vocab(seqs: List[str]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1, CLS: 2, MSK: 3}
    for s in seqs:
        for tok in s.strip().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"]["sequence"])
vocab_size = len(vocab)
pad_id, unk_id, cls_id, msk_id = vocab[PAD], vocab[UNK], vocab[CLS], vocab[MSK]
print(f"Vocab size: {vocab_size}")


def encode(seq: str, max_len: int) -> List[int]:
    tokens = [CLS] + seq.strip().split()
    ids = [vocab.get(tok, unk_id) for tok in tokens][:max_len]
    ids += [pad_id] * (max_len - len(ids))
    return ids


max_len = min(
    max(len(s.split()) for s in spr["train"]["sequence"]) + 1, 64
)  # +1 for CLS
print(f"max_len: {max_len}")

# --- labels ---------------------------------------------------------------------------
label_set = sorted(set(spr["train"]["label"]))
label2id = {lab: i for i, lab in enumerate(label_set)}
num_labels = len(label2id)
print(f"num_labels: {num_labels}")


# --- dataset --------------------------------------------------------------------------
class SPRDataset(Dataset):
    def __init__(self, split, do_mlm: bool):
        self.seqs = split["sequence"]
        self.labels = [label2id[l] for l in split["label"]]
        self.do_mlm = do_mlm

    def random_mask(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        """BERT-style random masking (15% of tokens, excluding PAD & CLS)."""
        ids = ids.copy()
        labels = [-100] * len(ids)
        for i in range(1, len(ids)):  # skip CLS at position 0
            if ids[i] == pad_id:
                continue
            if random.random() < 0.15:
                labels[i] = ids[i]
                prob = random.random()
                if prob < 0.8:
                    ids[i] = msk_id
                elif prob < 0.9:
                    ids[i] = random.randrange(vocab_size)
                # 10% keep unchanged
        return ids, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        input_ids = encode(self.seqs[idx], max_len)
        mlm_labels = [-100] * max_len
        if self.do_mlm:
            input_ids, mlm_labels = self.random_mask(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
        }


batch_size = 64
train_loader = DataLoader(
    SPRDataset(spr["train"], do_mlm=True), batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    SPRDataset(spr["dev"], do_mlm=False), batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    SPRDataset(spr["test"], do_mlm=False), batch_size=batch_size, shuffle=False
)


# --- model ----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SPRTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, nlayers, num_labels, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.posenc = PositionalEncoding(emb_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls_head = nn.Linear(emb_dim, num_labels)
        self.mlm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, input_ids):
        mask = input_ids == pad_id
        x = self.embedding(input_ids)
        x = self.posenc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        cls_vec = x[:, 0, :]  # embedding of <cls>
        return self.cls_head(cls_vec), self.mlm_head(x)


# --- training / evaluation ------------------------------------------------------------
def run_epoch(
    model, loader, criterion_cls, criterion_mlm, optimizer=None, mlm_weight=0.5
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, total_cls_loss, total_mlm_loss = 0, 0, 0
    all_preds, all_trues = [], []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            cls_logits, mlm_logits = model(batch["input_ids"])
            cls_loss = criterion_cls(cls_logits, batch["labels"])
            mlm_loss = torch.tensor(0.0, device=device)
            if train_mode:  # MLM used only during training
                mlm_loss = criterion_mlm(
                    mlm_logits.view(-1, vocab_size), batch["mlm_labels"].view(-1)
                )
            loss = cls_loss + mlm_weight * mlm_loss
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)
        total_cls_loss += cls_loss.item() * batch["labels"].size(0)
        total_mlm_loss += mlm_loss.item() * batch["labels"].size(0)
        all_preds.extend(cls_logits.argmax(dim=-1).cpu().numpy())
        all_trues.extend(batch["labels"].cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    return avg_loss, macro_f1


# --- hyper-params ---------------------------------------------------------------------
emb_dim = 128
nhead = 8
nlayers = 2
dropout = 0.1
lr = 1e-3
num_epochs = 10
mlm_weight = 0.5

# --- initialise -----------------------------------------------------------------------
model = SPRTransformer(vocab_size, emb_dim, nhead, nlayers, num_labels, dropout).to(
    device
)
criterion_cls = nn.CrossEntropyLoss()
criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- train / validate -----------------------------------------------------------------
record = experiment_data["SPR_BENCH_MLM"]
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_loss, train_f1 = run_epoch(
        model, train_loader, criterion_cls, criterion_mlm, optimizer, mlm_weight
    )
    val_loss, val_f1 = run_epoch(
        model, val_loader, criterion_cls, criterion_mlm, optimizer=None
    )
    record["losses"]["train"].append(train_loss)
    record["losses"]["val"].append(val_loss)
    record["metrics"]["train_macro_f1"].append(train_f1)
    record["metrics"]["val_macro_f1"].append(val_f1)
    record["epochs"].append(epoch)
    print(
        f"Epoch {epoch}: val_loss = {val_loss:.4f}, val_macro_f1 = {val_f1:.4f} "
        f"(train_macro_f1 = {train_f1:.4f}, time {time.time()-t0:.1f}s)"
    )

# --- test -----------------------------------------------------------------------------
test_loss, test_f1 = run_epoch(
    model, test_loader, criterion_cls, criterion_mlm, optimizer=None
)
print(f"Test: loss = {test_loss:.4f}, macro_F1 = {test_f1:.4f}")
record["test_loss"] = test_loss
record["test_macro_f1"] = test_f1

# --- save -----------------------------------------------------------------------------
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Results saved to", os.path.join(working_dir, "experiment_data.npy"))
