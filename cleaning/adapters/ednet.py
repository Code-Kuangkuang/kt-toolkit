"""Both EdNet slices, since they differ only in which users they take.

`process_raw_data` returns the rewritten output directory for this dataset --
the preprocessor redirects from the shared KT1 tree into data/<slice>/ -- so the
split steps are given that rather than the input path.
"""

from preprocess.data_proprocess import process_raw_data
from preprocess.split_datasets import main as split_concept
from preprocess.split_datasets_que import main as split_question


def _run(cfg):
    name = cfg["dataset_name"]
    dname2paths = {name: cfg["raw_path"]}
    dname, writef = process_raw_data(name, dname2paths)

    split_concept(
        dname,
        writef,
        name,
        cfg["configf"],
        cfg["min_seq_len"],
        cfg["maxlen"],
        cfg["kfold"],
    )
    if cfg.get("gen_question_level", True):
        split_question(
            dname,
            writef,
            name,
            cfg["configf"],
            cfg["min_seq_len"],
            cfg["maxlen"],
            cfg["kfold"],
        )


def ednet(cfg):
    return _run(cfg)


def ednet5w(cfg):
    return _run(cfg)
