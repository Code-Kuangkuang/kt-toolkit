from preprocess.data_proprocess import process_raw_data
from preprocess.split_datasets import main as split_concept
from preprocess.split_datasets_que import main as split_question

def run(cfg):
    dname2paths = {"assist2009": cfg["raw_path"]}
    dname, writef = process_raw_data(cfg["dataset_name"], dname2paths)

    split_concept(dname, writef, cfg["dataset_name"], cfg["configf"],
                  cfg["min_seq_len"], cfg["maxlen"], cfg["kfold"])
    if cfg.get("gen_question_level", True):
        split_question(dname, writef, cfg["dataset_name"], cfg["configf"],
                       cfg["min_seq_len"], cfg["maxlen"], cfg["kfold"])