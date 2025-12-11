import multiprocessing as mp
import pandas as pd
import os
import sqlite3
import numpy as np
import json
from tokenizers import Tokenizer
from vocabulary_functions import get_family_dictionary
from pathlib import Path
from statistics import mean, stdev
from Bio.Align import substitution_matrices

base_path = Path("/cta/share/users/ProteinGym/DMS_Assays(Substitutions)")
df = pd.concat((pd.read_csv(file) for file in base_path.rglob("*.csv") if "HUMAN" in str(file)), ignore_index=True)
df = df[~df["mutant"].str.contains(":")].copy() # count only single muts
df = df[~((df["DMS_score"] > 5) | (df["DMS_score"] < -5))] # elim outliers
df["mutations"] = df["mutant"].apply(lambda x: (x[0], int(x[1:-1]) - 1, x[-1]))
def reverse_mutation(mut_seq, mut_data):
    mut_o, mut_p, mut_m = mut_data
    return mut_seq[:mut_p] + mut_o + mut_seq[mut_p+1:]
df["ori_seq"] = df.apply(lambda row: reverse_mutation(row["mutated_sequence"], row["mutations"]), axis=1)

def replace_at_i(s, t, i):
    return s[:i] + t + s[i+1:]


def slide_find_tokens(target_pos, seq, max_len = 10):
    '''
    Finds all the potential tokens, in sequence seq, 
    around the position target_pos, up to length max_len.
    
    Parameters
    ----------
    target_pos : position to search around
    seq : sequence to search
    max_len : maximum length for token search.

    Returns
    -------
    list
        the list of possible tokens.
    '''
    possible_tokens = []

    for cur_len in range(1, max_len + 1):
        for i in range(cur_len):
            tk = seq[target_pos - i:target_pos+cur_len - i]
            if len(tk) < cur_len:
                continue
            possible_tokens.append(tk)
    return possible_tokens


jobs = []
job_id = 0
for mat in ["blosum62", "pam250", "blosum45", "pam70"]:
    vocabs = []
    base_path = f"/cta/share/users/mutbpe/tokenizers/{mat}/"
    names = os.listdir(base_path)
    names = list(filter(lambda x: x.startswith("hf_uniref50_") and "512000" not in x, names))
    # ignore BPE
    names = list(filter(lambda x: "_bpe_" not in x, names))

    tkzs = [base_path + p for p in names]
    vocabs = [base_path + p[3:] for p in names]
    for tkz, vocab in zip(tkzs, vocabs):
        job = (job_id, df, mat, (tkz, vocab))
        jobs.append(job)
        job_id += 1

def worker(data):
    is_segmented = True
    job_id, df, mat, (tkz_path, vocab_path) = data 
    print(f"Starting on job {job_id}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tkz = Tokenizer.from_file(tkz_path)
    with open(vocab_path) as f:
        vocab = json.load(f)
    families = get_family_dictionary(vocab)
    same_count_pos = 0
    same_count_neg = 0
    sibling_count_pos = 0
    sibling_count_neg = 0
    for (ori_seq,), group in df.groupby(["ori_seq"]):
        mut_dict = {}
        for (_, pos, t), sc, mut_seq in zip(group["mutations"], group["DMS_score_bin"], group["mutated_sequence"]):
            mut_dict.setdefault(pos, [])
            mut_dict[pos].append((t, sc, mut_seq))
        ori_enc = tkz.encode(ori_seq)
        for pos, mut_l in mut_dict.items():
            for i, (st, en) in enumerate(ori_enc.offsets):
                if st <= pos and pos < en:
                    tk_pos = i
                    break
            ori_tk = ori_enc.tokens[tk_pos]
            for (t, sc, mut_seq) in mut_l:
                mut_enc = tkz.encode(mut_seq)
                if mut_enc.offsets == ori_enc.offsets: # SAME case
                    if sc == 1: same_count_pos += 1
                    else: same_count_neg += 1
                    mut_tk = mut_enc.tokens[tk_pos]
                    if ori_tk in families[mut_tk]: # Sibling case
                        if sc == 1: sibling_count_pos += 1
                        else: sibling_count_neg += 1

        

    return (job_id, mat, tkz_path, same_count_pos, same_count_neg, sibling_count_pos, sibling_count_neg)

def main():
    with mp.Pool(processes=50) as pool:
        results = pool.map(worker, jobs)
    results.sort()
    df = pd.DataFrame(results, columns=["job_id", "mat", "tkz_path", "same_count_pos", "same_count_neg", "sibling_count_pos", "sibling_count_neg"])
    print(df.head())
    df.to_csv("dms_analysis/results.csv")
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
    df.to_csv("dms_analysis/results_processed.csv")

if __name__ == "__main__":
    main()