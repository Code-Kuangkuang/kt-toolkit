import pandas as pd
from .utils import sta_infos, write_txt, format_list2str

keys = ["studentId", "skill", "problemId"]


def read_data_from_csv(read_file, write_file, keep_scaffolding=False):
    """Build assist2017 sequences.

    keep_scaffolding=True reproduces the unfiltered action log (the commonly
    cited 1,709 students / 942,816 interactions protocol). It keeps BOTH
    scaffolding rows and repeated attempt actions, because both are needed to
    reach that interaction count. Use it only to compare against papers that
    report those numbers; it leaves the label leakage described below in place.
    """
    df = pd.read_csv(read_file, encoding='utf-8', low_memory=False)

    stares = []
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, keys, stares)
    print(
        f"original interaction num: {df.shape[0]}, user num: {df['studentId'].nunique()}, question num: {df['problemId'].nunique()}, "
        f"concept num: {df['skill'].nunique()}, avg(ins) per s:{avgins}, avg(c) per q:{avgcq}, na:{na}")

    df["index"] = range(len(df))

    df = df.dropna(subset=["studentId", "problemId", "correct", "skill", "startTime"])
    df = df[df['correct'].isin([0, 1])]
    # This release is an ACTION log: one row per attempt or help request, not per
    # problem. Two structures leak the label if kept.
    #   1. Scaffolding sub-problems (original == 0) are generated only after a
    #      wrong answer on the main problem.
    #   2. Within original == 1 a failed problem is retried until solved, so the
    #      same problem repeats consecutively with labels 0,0,...,1.
    # Keep main problems only, then collapse each attempt sequence to its first
    # action, whose `correct` is first-attempt correctness. This matches the
    # granularity of the assist2009 "collapsed" release (one row per problem).
    if not keep_scaffolding:
        df = df[df['original'] == 1]
        df = df.sort_values(by=['startTime', 'index'], kind='mergesort')
        df = df.drop_duplicates(subset=['studentId', 'problemId', 'assignmentId'], keep='first')
    df.loc[:, 'timeTaken'] = df['timeTaken'].apply(lambda x: round(x * 1000))

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, keys, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df2 = df[["index", "studentId", "problemId", "skill", "correct", "timeTaken", "startTime"]]
    ui_df = df2.groupby('studentId', sort=False)

    user_inter = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]  
        tmp_inter.loc[:, 'startTime'] = tmp_inter.loc[:, 'startTime'].apply(lambda t: int(t) * 1000)
        tmp_inter = tmp_inter.sort_values(by=['startTime', 'index'])

        tmp_inter['startTime'] = tmp_inter['startTime']

        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['problemId'].tolist()
        seq_skills = tmp_inter['skill'].tolist()
        seq_ans = tmp_inter['correct'].tolist()
        seq_submit_time = tmp_inter['startTime'].tolist()
        seq_response_cost = tmp_inter['timeTaken'].tolist()

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_submit_time) == len(seq_response_cost)

        user_inter.append(
            [[str(user), str(seq_len)], format_list2str(seq_problems), seq_skills, format_list2str(seq_ans), format_list2str(seq_submit_time), format_list2str(seq_response_cost)])

    write_txt(write_file, user_inter)


