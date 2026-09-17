import os

import numpy as np
import pandas as pd


def _read_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def generate_time2idx(data_config, folds=None):
    dpath = data_config["dpath"]
    train_name = data_config.get("train_valid_original_file", "train_valid.csv")
    test_name = data_config.get("test_original_file", "test.csv")

    frames = []
    # HDKT passes the current training folds so categorical time vocabularies
    # are fitted without validation/test interactions.  The optional argument
    # preserves the historical LPKT behavior for existing callers.
    names = (train_name,) if folds is not None else (train_name, test_name)
    fold_set = None if folds is None else {int(fold) for fold in folds}
    for name in names:
        df = _read_csv_if_exists(os.path.join(dpath, name))
        if df is not None:
            if fold_set is not None and "fold" in df.columns:
                fold_values = pd.to_numeric(df["fold"], errors="raise").astype(int)
                df = df.loc[fold_values.isin(fold_set)]
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"Cannot generate LPKT time indices: no train/test csv found in {dpath}."
        )

    df = pd.concat(frames, ignore_index=True)
    at2idx = {}
    it2idx = {}

    for _, row in df.iterrows():
        if "usetimes" in row.index and row["usetimes"]:
            use_times = [int(float(t)) // 1000 for t in str(row["usetimes"]).split(",") if t != ""]
            for at in use_times:
                key = str(at)
                if key not in at2idx:
                    at2idx[key] = len(at2idx)

        if "timestamps" in row.index and row["timestamps"]:
            timestamps = [int(float(t)) for t in str(row["timestamps"]).split(",") if t != ""]
            if timestamps:
                shft_timestamps = timestamps[:1] + timestamps[:-1]
                it = np.maximum(
                    np.minimum(
                        (np.array(timestamps) - np.array(shft_timestamps)) // 1000 // 60,
                        43200,
                    ),
                    -1,
                )
                for t in it:
                    key = str(int(t))
                    if key not in it2idx:
                        it2idx[key] = len(it2idx)
        else:
            if "1" not in it2idx:
                it2idx["1"] = len(it2idx)

    at2idx["-1"] = len(at2idx)
    it2idx["-1"] = len(it2idx)
    return at2idx, it2idx
