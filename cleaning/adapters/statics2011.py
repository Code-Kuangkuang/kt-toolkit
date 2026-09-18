from preprocess.data_proprocess import process_raw_data
from preprocess.split_datasets import main as split_concept


def run(cfg):
    dname2paths = {"statics2011": cfg["raw_path"]}
    dname, writef = process_raw_data(cfg["dataset_name"], dname2paths)

    split_concept(
        dname,
        writef,
        cfg["dataset_name"],
        cfg["configf"],
        cfg["min_seq_len"],
        cfg["maxlen"],
        cfg["kfold"],
    )
    # No question-level split: the preprocessor joins `Problem Name----Step Name`
    # into the concept id, so there is no separate question to group by. Calling
    # split_question here would write files the config does not declare and the
    # loaders would never read.
