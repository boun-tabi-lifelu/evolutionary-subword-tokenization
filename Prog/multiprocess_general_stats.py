import multiprocessing as mp
import sqlite3, json
import pandas as pd
import bpe_functions, vocabulary_functions
import os
from Bio.Align import substitution_matrices
from tokenizers import Tokenizer
from collections import Counter
import statistics


db_file = "/cta/share/users/uniprot/human/human.db"
conn = sqlite3.connect(db_file)
df_uniprot_human_seqs = pd.read_sql(f"SELECT Sequence FROM proteins WHERE Entry IN (SELECT uniprot_accession FROM uniref50_distilled)", conn)
conn.close()
filtered_sequences = df_uniprot_human_seqs[
    (df_uniprot_human_seqs["Sequence"].str.count("X") <= 1) &
    (df_uniprot_human_seqs["Sequence"].str.count("B") <= 1) &
    (df_uniprot_human_seqs["Sequence"].str.count("U") <= 1) &
    (df_uniprot_human_seqs["Sequence"].str.count("Z") <= 1) &
    (df_uniprot_human_seqs["Sequence"].str.len() <= 3000)
]["Sequence"].tolist()

jobs = []
job_id = 0
for mat in ["blosum62", "pam250", "blosum45", "pam70"]:
    vocabs = []
    base_path = f"/cta/share/users/mutbpe/tokenizers/{mat}/"
    names = os.listdir(base_path)
    names = list(filter(lambda x: x.startswith("hf_uniref50_") and "512000" not in x, names))

    # count BPE only once
    if mat != "blosum62":
        names = list(filter(lambda x: "_bpe_" not in x, names))
    tkzs = [base_path + p for p in names]
    vocabs = [base_path + p[3:] for p in names]
    for tkz, vocab in zip(tkzs, vocabs):
        job = (job_id, filtered_sequences, mat, (tkz, vocab))
        jobs.append(job)
        job_id += 1

def worker(data):
    job_id, seqs, mat, (tkz_path, vocab_path) = data 
    print(f"Starting on job {job_id}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tkz = Tokenizer.from_file(tkz_path)
    with open(vocab_path) as f:
        vocab = json.load(f)

    v_size = len(vocab)
    parents = set(vocabulary_functions.get_parents(vocab))
    p_size = len(parents)
    mutateds = set(vocabulary_functions.get_mutated(vocab))
    m_size = len(mutateds)

    # Stats
    p_ratio = p_size / v_size
    m_ratio = m_size / v_size
    ulengths = [len(u) for u in list(vocab.keys())]

    ulength_mean = statistics.mean(ulengths)
    ulength_std = statistics.stdev(ulengths)

    encs = tkz.encode_batch(seqs)
    used_tokens = []
    for enc in encs:
        used_tokens += enc.tokens

    used_token_lengths = [len(t) for t in used_tokens]
    used_ulength_mean = statistics.mean(used_token_lengths)
    used_ulength_std = statistics.stdev(used_token_lengths)

    t_counts = Counter(used_tokens)
    count_list = list(t_counts.items())
    count_list = list(filter(lambda x: x[1] > 10, count_list))
    util10 = len(count_list) / v_size
    count_list = list(filter(lambda x: x[1] > 30, count_list))
    util30 = len(count_list) / v_size
    count_list = list(filter(lambda x: x[1] > 50, count_list))
    util50 = len(count_list) / v_size

    used_p = 0
    used_m = 0
    for t in used_tokens:
        if t in parents:
            used_p += 1
        if t in mutateds:
            used_m += 1

    used_p_ratio = used_p / len(used_tokens)
    used_m_ratio = used_m / len(used_tokens)

    return (job_id, mat, tkz_path, p_ratio, m_ratio, ulength_mean, ulength_std, util10, util30, util50, used_ulength_mean, used_ulength_std, used_p_ratio, used_m_ratio)

def main():
    with mp.Pool(processes=20) as pool:
        results = pool.map(worker, jobs)
    results.sort()
    df = pd.DataFrame(results, columns=["job_id", "mat", "tkz_path", "p_ratio", "m_ratio", "ulength_mean", "ulength_std", "util10", "util30", "util50", "used_ulength_mean", "used_ulength_std", "used_p_ratio", "used_m_ratio"])
    print(df.head())
    df.to_csv("general_stats/results.csv")
    # process table
    df["tkz_path"] = df["tkz_path"].apply(lambda x: x[x.find("hf"):])
    df["is_bpe"] = df["tkz_path"].str.contains("_bpe_")
    df["v_size"] = df["tkz_path"].apply(lambda x: x.split("_")[-1][:-5])
    df["freq_cutoff"] = df[~df["is_bpe"]]["tkz_path"].apply(lambda x: x.split("_")[-2])
    df["sim_cutoff"] = df[~df["is_bpe"]]["tkz_path"].apply(lambda x: x.split("_")[-5])
    cols = list(df.columns)
    newcols = cols[:3] + cols[-4:] + cols[3:-4]
    df = df[newcols]
    df.drop(columns=['job_id'])
    df.to_csv("general_stats/results_processed.csv")

if __name__ == "__main__":
    main()