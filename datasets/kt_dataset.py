import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class KTDataset(Dataset):
    def __init__(self, file_path, input_type, folds, pad_val=-1, use_timestamps=False):
        super().__init__()
        self.input_type = input_type
        self.pad_val = pad_val
        self.use_timestamps = use_timestamps
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(f) for f in folds])
        cache_tag = "ts1" if self.use_timestamps else "ts0"
        processed = file_path + folds_str + f"_{cache_tag}.pkl"

        if not os.path.exists(processed):
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)
        else:
            self.dori = pd.read_pickle(processed)

        if self.use_timestamps and "tseqs" not in self.dori:
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        dcur = {}
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            # Skip empty tensors/lists to avoid IndexError
            val = self.dori[key]
            if isinstance(val, list) and len(val) == 0:
                continue
            if hasattr(val, 'numel') and val.numel() == 0:
                continue
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        return dcur

    def _load_data(self, sequence_path, folds):
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "smasks": [], "itseqs": []}
        # Load timestamps if available and requested
        if self.use_timestamps:
            dori["tseqs"] = []
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
            # Load timestamps if available
            if self.use_timestamps and "timestamps" in row:
                dori["tseqs"].append([int(x) for x in row["timestamps"].split(",")])
                # Calculate itseqs (interaction interval in minutes, bucketed, same as pykt)
                timestamps = dori["tseqs"][-1]
                shft_timestamps = [0] + timestamps[:-1]
                it = [max(min((t - s) // 1000 // 60, 43200), -1) for t, s in zip(timestamps, shft_timestamps)]
                dori["itseqs"].append(it)

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

        # Convert timestamps to tensor if loaded
        if self.use_timestamps and "tseqs" in dori and len(dori["tseqs"]) > 0:
            dori["tseqs"] = torch.tensor(dori["tseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["tseqs"] = torch.tensor([])

        # Convert itseqs to tensor if loaded
        if self.use_timestamps and "itseqs" in dori and len(dori["itseqs"]) > 0:
            dori["itseqs"] = torch.tensor(dori["itseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["itseqs"] = torch.tensor([], dtype=torch.long)

        dori["masks"] = (seq_for_mask[:, :-1] != self.pad_val) & (
            seq_for_mask[:, 1:] != self.pad_val
        )
        dori["smasks"] = dori["smasks"][:, 1:] != self.pad_val
        return dori


class KTQueDataset(Dataset):
    """Question-level dataset for ALL-in-One mode.

    Supports 2D concept sequences [batch, seq_len, max_concepts].
    Each concept position can contain multiple concepts separated by "_".
    """
    def __init__(
        self,
        file_path,
        input_type,
        folds,
        concept_num,
        max_concepts=4,
        pad_val=-1,
        concept_mode: str = "multi",
        use_timestamps=False,
    ):
        super().__init__()
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        self.pad_val = pad_val
        self.concept_mode = concept_mode
        self.use_timestamps = use_timestamps
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(f) for f in folds])
        cache_tag = "ts1" if self.use_timestamps else "ts0"
        processed = file_path + folds_str + f"_{cache_tag}_qlevel.pkl"

        if not os.path.exists(processed):
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)
        else:
            self.dori = pd.read_pickle(processed)

        if self.use_timestamps and "tseqs" not in self.dori:
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        dcur = {}
        mseqs = self.dori["masks"][index]

        def _apply_time_mask(x, m):
            mask = m
            while mask.dim() < x.dim():
                mask = mask.unsqueeze(-1)
            return x * mask

        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            # Skip empty tensors/lists to avoid IndexError
            val = self.dori[key]
            if isinstance(val, list) and len(val) == 0:
                continue
            if hasattr(val, 'numel') and val.numel() == 0:
                continue

            cur = self.dori[key][index]
            if key == "cseqs" and cur.dim() >= 2 and self.concept_mode == "first":
                cur = cur[:, 0]

            seqs = _apply_time_mask(cur[:-1], mseqs)
            shft_seqs = _apply_time_mask(cur[1:], mseqs)
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        return dcur

    def _load_data(self, sequence_path, folds):
        """Load data with 2D concept sequences.

        Concepts are stored as [seq_len, max_concepts] 2D array.
        Format: "1_2_3_-1" means concepts [1,2,3,-1] at one position.
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "smasks": [], "itseqs": []}
        # Load timestamps if available and requested
        if self.use_timestamps:
            dori["tseqs"] = []
        df = pd.read_csv(sequence_path, dtype=str, keep_default_na=False)
        if "fold" in df.columns:
            df["fold"] = df["fold"].astype(int)
        df = df[df["fold"].isin(folds)]

        for _, row in df.iterrows():
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills + [-1] * (self.max_concepts - len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)  # 2D: [seq_len, max_concepts]
            if "questions" in self.input_type:
                dori["qseqs"].append([int(x) for x in row["questions"].split(",")])
            dori["rseqs"].append([int(x) for x in row["responses"].split(",")])
            dori["smasks"].append([int(x) for x in row["selectmasks"].split(",")])
            # Load timestamps if available
            if self.use_timestamps and "timestamps" in row:
                dori["tseqs"].append([int(x) for x in row["timestamps"].split(",")])
                # Calculate itseqs (interaction interval in minutes, bucketed, same as pykt)
                timestamps = dori["tseqs"][-1]
                shft_timestamps = [0] + timestamps[:-1]
                it = [max(min((t - s) // 1000 // 60, 43200), -1) for t, s in zip(timestamps, shft_timestamps)]
                dori["itseqs"].append(it)

        # Convert to tensors
        if len(dori["cseqs"]) > 0:
            # cseqs is 2D: [num_samples, seq_len, max_concepts]
            dori["cseqs"] = torch.tensor(dori["cseqs"], dtype=torch.long)
            seq_for_mask = dori["cseqs"][:, :, 0]  # Use first concept for masking
        elif len(dori["qseqs"]) > 0:
            seq_for_mask = torch.tensor(dori["qseqs"], dtype=torch.long)
        else:
            raise ValueError("No concepts/questions found in sequence file.")

        if len(dori["qseqs"]) > 0:
            dori["qseqs"] = torch.tensor(dori["qseqs"], dtype=torch.long)
        else:
            dori["qseqs"] = torch.tensor([])

        dori["rseqs"] = torch.tensor(dori["rseqs"], dtype=torch.float)
        dori["smasks"] = torch.tensor(dori["smasks"], dtype=torch.long)

        # Convert timestamps to tensor if loaded
        if self.use_timestamps and "tseqs" in dori and len(dori["tseqs"]) > 0:
            dori["tseqs"] = torch.tensor(dori["tseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["tseqs"] = torch.tensor([])

        # Convert itseqs to tensor if loaded (for KTQueDataset)
        if self.use_timestamps and "itseqs" in dori and len(dori["itseqs"]) > 0:
            dori["itseqs"] = torch.tensor(dori["itseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["itseqs"] = torch.tensor([], dtype=torch.long)

        # masks based on first concept column
        dori["masks"] = (seq_for_mask[:, :-1] != self.pad_val) & (
            seq_for_mask[:, 1:] != self.pad_val
        )
        dori["smasks"] = dori["smasks"][:, 1:] != self.pad_val
        return dori
