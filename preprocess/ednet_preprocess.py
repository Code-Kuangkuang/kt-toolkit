import pandas as pd
import random
import os
from .utils import sta_infos, write_txt
from tqdm import tqdm

KEYS = ["user_id", "tags", "question_id"]


# `ednet` and `ednet5w` are not two datasets. Both are deterministic samples of
# EdNet-KT1, cut from one shuffle of the user ids under a fixed seed:
#
#   ednet     the first 5,000 users encountered
#   ednet5w   the next 50,000, skipping those same 5,000
#
# Same seed, so the two slices are disjoint and reproducible, and neither is the
# full dataset -- KT1 holds 784,309 users. A paper saying "we use EdNet" has said
# almost nothing; SAMPLING below is recorded into data_config.json so the
# distinction survives into the artifacts rather than living in someone's memory.
#
# The id range scanned is 840,473 while only 784,309 files exist, so roughly
# 56,000 ids have no file. That is why membership is decided by os.path.exists
# and why "the first 5,000" means the first 5,000 *found*, not the first 5,000
# shuffled ids.
SAMPLING = {
    "ednet": {"seed": 2, "skip_users": 0, "take_users": 5000},
    "ednet5w": {"seed": 2, "skip_users": 5000, "take_users": 50000},
}
ID_RANGE = 840473


def read_data_from_csv(read_file, write_file, dataset_name=None):
    if dataset_name not in SAMPLING:
        raise ValueError(
            f"EdNet preprocessing needs dataset_name to be one of "
            f"{sorted(SAMPLING)}, got {dataset_name!r}. The name selects which "
            "slice of KT1 to take; there is no unsliced 'ednet'."
        )
    plan = SAMPLING[dataset_name]
    wanted = plan["skip_users"] + plan["take_users"]

    write_file = write_file.replace("/ednet/", f"/{dataset_name}/")
    write_dir = read_file.replace("/ednet/", f"/{dataset_name}")
    print(f"write_dir is {write_dir}")
    print(f"write_file is {write_file}")

    contents_path = os.path.join(read_file, 'contents', 'questions.csv')
    if not os.path.exists(contents_path):
        # Checked before the scan rather than after: the loop below walks 840k
        # ids and takes many minutes, and KT1 carries no correct_answer or tags
        # of its own, so without this file nothing downstream can be computed.
        raise FileNotFoundError(
            f"EdNet question metadata not found at {contents_path}. KT1 records "
            "only timestamp/solving_id/question_id/user_answer/elapsed_time, so "
            "the concepts (`tags`) and the answer key (`correct_answer`) both "
            "come from here. Download EdNet-Contents.zip (174 KB) from "
            "http://base.ustc.edu.cn/data/EdNet/ and extract it so that "
            f"{os.path.join(read_file, 'contents')} exists."
        )

    stares = []
    file_list = list()

    random.seed(plan["seed"])
    samp = [i for i in range(ID_RANGE)]
    random.shuffle(samp)

    count = 0

    for unum in tqdm(samp):
        str_unum = str(unum)
        df_path = os.path.join(read_file, f"KT1/u{str_unum}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            df['user_id'] = unum

            file_list.append(df)
            count = count + 1

        if count == wanted:
            break

    print(f"total user num: {count}")
    if count < wanted:
        # `start_i` used to be assigned only inside the break branches, so a
        # short KT1 fell out of the loop and died on a NameError at the concat
        # below -- after walking all 840k ids, with a message naming neither the
        # cause nor the fix.
        raise ValueError(
            f"{dataset_name} needs {wanted:,} users ({plan['skip_users']:,} "
            f"skipped + {plan['take_users']:,} taken) but only {count:,} KT1 "
            f"files were found under {os.path.join(read_file, 'KT1')}. "
            "The full KT1 release holds 784,309; extract all of it."
        )

    all_sa = pd.concat(file_list[plan["skip_users"]:])
    print(f"after sub all_sa: {len(all_sa)}")
    all_sa["index"] = range(all_sa.shape[0])
    ca = pd.read_csv(contents_path)
    
    # From here on, write to the slice's own directory rather than back into the
    # shared KT1 tree.
    read_file = write_dir

    all_sa.to_csv(os.path.join(read_file, 'ednet_sample.csv'), index=False)
    ca['tags'] = ca['tags'].apply(lambda x:x.replace(";","_"))
    ca = ca[ca['tags']!='-1']
    co = all_sa.merge(ca, sort=False,how='left')
    co = co.dropna(subset=["user_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
    co['correct'] = (co['correct_answer']==co['user_answer']).apply(int)


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(co, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(co, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    co.to_csv(os.path.join(read_file, 'ednet_sample_process.csv'), index=False)
    
    ui_df = co.groupby('user_id', sort=False)

    user_inters = []
    for ui in tqdm(ui_df):
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["timestamp", "index"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter['tags'].astype(str)
        seq_ans = tmp_inter['correct'].astype(str)
        seq_problems = tmp_inter['question_id'].astype(str)
        seq_start_time = tmp_inter['timestamp'].astype(str)
        seq_response_cost = tmp_inter['elapsed_time'].astype(str)

        assert seq_len == len(seq_problems) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)
    print("\n".join(stares))
    return write_dir, write_file


