import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class KTDataset(Dataset):
    def __init__(self, file_path, input_type, folds, pad_val=-1):
        super().__init__()
        self.input_type = input_type
        self.pad_val = pad_val
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(f) for f in folds])
        processed = file_path + folds_str + ".pkl"

        if not os.path.exists(processed):
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)
        else:
            self.dori = pd.read_pickle(processed)

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        dcur = {}
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        return dcur

    def _load_data(self, sequence_path, folds):
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "smasks": []}
        df = pd.read_csv(sequence_path, dtype=str, keep_default_na=False)
        if "fold" in df.columns:
            df["fold"] = df["fold"].astype(int)
        df = df[df["fold"].isin(folds)]

        for _, row in df.iterrows():
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(x) for x in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(x) for x in row["questions"].split(",")])
            dori["rseqs"].append([int(x) for x in row["responses"].split(",")])
            dori["smasks"].append([int(x) for x in row["selectmasks"].split(",")])

        if len(dori["cseqs"]) > 0:
            seq_for_mask = torch.tensor(dori["cseqs"], dtype=torch.long)
        elif len(dori["qseqs"]) > 0:
            seq_for_mask = torch.tensor(dori["qseqs"], dtype=torch.long)
        else:
            raise ValueError("No concepts/questions found in sequence file.")

        if len(dori["cseqs"]) > 0:
            dori["cseqs"] = torch.tensor(dori["cseqs"], dtype=torch.long)
        else:
            dori["cseqs"] = torch.tensor([])
        if len(dori["qseqs"]) > 0:
            dori["qseqs"] = torch.tensor(dori["qseqs"], dtype=torch.long)
        else:
            dori["qseqs"] = torch.tensor([])
        dori["rseqs"] = torch.tensor(dori["rseqs"], dtype=torch.float)
        dori["smasks"] = torch.tensor(dori["smasks"], dtype=torch.long)

        dori["masks"] = (seq_for_mask[:, :-1] != self.pad_val) & (
            seq_for_mask[:, 1:] != self.pad_val
        )
        dori["smasks"] = dori["smasks"][:, 1:] != self.pad_val
        return dori
