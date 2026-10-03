"""Prepare the labeled training release; official hidden test labels stay untouched."""
from preprocess.data_proprocess import process_raw_data
from preprocess.split_datasets import main as split_concept
from preprocess.split_datasets_que import main as split_question


def run(cfg):
    name = cfg["dataset_name"]
    dname, writef = process_raw_data(name, {name: cfg["raw_path"]})
    split_concept(dname, writef, name, cfg["configf"],
                  cfg["min_seq_len"], cfg["maxlen"], cfg["kfold"])
    if cfg.get("gen_question_level", True):
        split_question(dname, writef, name, cfg["configf"],
                       cfg["min_seq_len"], cfg["maxlen"], cfg["kfold"])
