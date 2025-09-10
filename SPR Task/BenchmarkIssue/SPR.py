"""
SPR.py
────────────────────────────────────────────────────────
Utility to load the 20 SPR benchmark datasets
Using HuggingFace’s `datasets` library.

Directory layout expected
SPR_BENCH/
 ├─ SFRFG/
 │   ├─ train.csv
 │   ├─ dev.csv
 │   └─ test.csv
 ├─ IJSJF/
 └─ …

Each CSV has header:  id,sequence,label
────────────────────────────────────────────────────────
$ pip install datasets   # once
$ python read_spr_bench.py  --root SPR_BENCH
"""
import argparse
import pathlib
from typing import Dict

from datasets import load_dataset, DatasetDict                                         # <- no pandas import


def load_single_benchmark(bdir: pathlib.Path) -> DatasetDict:
    """
    Return a DatasetDict {'train':…, 'dev':…, 'test':…} for one SPR ID folder.
    """
    def _load(split_csv: str):
        return load_dataset(
            "csv",
            data_files=str(bdir / split_csv),
            split="train",           # treat csv as a single split
            cache_dir=".cache_dsets" # optional; keeps HF cache tidy
        )

    dset = DatasetDict()
    dset["train"] = _load("train.csv")
    dset["dev"]   = _load("dev.csv")
    dset["test"]  = _load("test.csv")
    return dset


def load_spr_bench(root: pathlib.Path) -> Dict[str, DatasetDict]:
    """
    Walk SPR_BENCH/  →  {"SFRFG": DatasetDict, … "EWERV": DatasetDict}
    """
    benchmarks: Dict[str, DatasetDict] = {}
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            print(f"Loading {sub.name}")
            benchmarks[sub.name] = load_single_benchmark(sub)
    return benchmarks


def main():

    ## Absolute path of the datasets
    DATA_PATH = pathlib.Path('/home/zxl240011/AI-Scientist-v2/SPR_BENCH/')
    spr_bench = load_spr_bench(DATA_PATH)

    print("Benchmarks names:", spr_bench.keys())

    # Demo: show first example from IJSJF‑train
    ex = spr_bench["IJSJF"]["train"][0]
    print("\nExample row:")
    print(ex)          


if __name__ == "__main__":
    main()
