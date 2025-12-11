import pandas as pd
from pathlib import Path
import os
import re
import pickle
import json
from vocabulary_functions import get_family_dictionary
from tokenizers import Tokenizer
import random
import multiprocessing as mp
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Given a dataframe, generate a random dataset based on it.
def generate_random_mutations(df_original, seed = 25):
    # List of all 20 amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    random.seed(seed)
    df = df_original.copy()
    df["mutant"] = ""
    df["mutated_sequence"] = ""

    for i, row in df.iterrows():
        protein_seq = row["protein_sequence"]
        if not protein_seq:
            continue

        # Randomly select a position (1-indexed for mutant string)
        pos = random.randint(1, len(protein_seq))
        original_aa = protein_seq[pos - 1]

        # Choose a different amino acid
        possible_mutants = [aa for aa in amino_acids if aa != original_aa]
        new_aa = random.choice(possible_mutants)

        # Create the mutant string like K232E
        mutant_str = f"{original_aa}{pos}{new_aa}"

        # Apply mutation to the sequence
        mutated_seq = protein_seq[:pos - 1] + new_aa + protein_seq[pos:]

        # Fill in the values
        df.at[i, "mutant"] = mutant_str
        df.at[i, "mutated_sequence"] = mutated_seq

    return df

# Load datasets
base_path = Path("/cta/share/users/ProteinGym/Clinical_Variants(Substitutions)")
df = pd.concat((pd.read_csv(file) for file in base_path.rglob("*.csv")), ignore_index=True)
df = df.drop(columns="Unnamed: 0")

df_benign = df[df["DMS_bin_score"] == "Benign"]
df_benign = df_benign.drop(columns="DMS_bin_score")
df_patho = df[df["DMS_bin_score"] == "Pathogenic"]
df_patho = df_patho.drop(columns="DMS_bin_score")



# If you have not generated RANDOM datasets yet:

# df_benign.to_pickle("same_idx_work/datasets/df_benign.pkl")
# df_patho.to_pickle("same_idx_work/datasets/df_pathogenic.pkl")
# for seed in range(1, 11):
#     df_rand = generate_random_mutations(df.drop(columns="DMS_bin_score"), seed=seed)
#     df_rand.to_pickle(f"same_idx_work/datasets/df_rand{seed}.pkl")


rand_datasets = []
for seed in range(1, 11):
    with open(f"same_idx_work/datasets/df_rand{seed}.pkl", "rb") as f:
        rand_datasets.append(pickle.load(f))




# Enumerate jobs:
import os
jobs = []
job_id = 0
for dataset in ["same_idx_work/datasets/df_benign.pkl", 
                "same_idx_work/datasets/df_pathogenic.pkl"] + \
                [f"same_idx_work/datasets/df_rand{i}.pkl" for i in range(1, 11)]:
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
            job = (job_id, dataset, mat, (tkz, vocab))
            jobs.append(job)
            job_id += 1
print("total_num_jobs:", len(jobs))





# Worker function
def worker(data):
    job_id, dataset, mat, (tkz_path, vocab_path) = data 

    tkz = Tokenizer.from_file(tkz_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(dataset, "rb") as f:
        cur_df = pickle.load(f)

    families = get_family_dictionary(vocab)
    seq_list = cur_df["protein_sequence"].to_list()
    mutseq_list = cur_df["mutated_sequence"].to_list()
    mut_list = cur_df["mutant"].to_list()
    mut_list = [(int(mut[1:-1]) - 1, mut[0], mut[-1]) for mut in mut_list]

    same_count = 0
    sibling_count = 0
    ori_encs = tkz.encode_batch(seq_list)
    mut_encs = tkz.encode_batch(mutseq_list)

    if "mutbpe" in tkz_path: # PUMA
        for i, mut in enumerate(mut_list):
            if ori_encs[i].offsets == mut_encs[i].offsets: # SAME cases
                same_count += 1
                pos, _, _ = mut
                cur_token_ori = ""
                cur_token_mut = ""
                for j, (st, en) in enumerate(ori_encs[i].offsets):
                    if pos >= st and pos < en:
                        cur_token_ori = ori_encs[i].tokens[j]
                        cur_token_mut = mut_encs[i].tokens[j]
                        break
                if cur_token_ori in families[cur_token_mut]:
                    sibling_count += 1
    else:
        for i, mut in enumerate(mut_list):
            if ori_encs[i].offsets == mut_encs[i].offsets: # SAME cases
                same_count += 1

    same_idx = same_count/len(mut_list)
    sibling_rate = sibling_count/same_count
    # print(f"process finished job{job_id}")
    if job_id % 25 == 0:
        print(f"process finished job{job_id}")
    return (job_id, dataset, mat, tkz_path, same_idx, sibling_rate)


def main():
    with mp.Pool(processes=20) as pool:
        results = pool.map(worker, jobs)
    results.sort()
    df = pd.DataFrame(results, columns=["job_id", "dataset", "matrix", "vocab", "same_idx", "same_sibling_rate"])
    print(df.head())
    df.to_csv("same_idx_work/results.csv")
    
    # process results for better format
    res_data = df
    # res_data = res_data.drop(columns="Unnamed: 0")
    res_data['vocab_size'] = res_data['vocab'].str.extract(r'_(\d+)\.json$')
    res_data["config"] = res_data['vocab'].str.extract(r'mutbpe_(.+)_(\d+)\.json$')[0].fillna("bpe")

    def f(x):
        if "benign" in x: return "Benign"
        elif "patho" in x: return "Pathogenic"
        else:
            rand_id = re.match(r"df_rand(\d+)\.pkl", x.split("/")[-1]).group(1)
            return f"Random {rand_id}"

    res_data["dataset"] = res_data["dataset"].apply(f)
    res_data["is_pre"] = res_data["vocab"].str.contains("uniref50pre")
    res_data = res_data.drop(columns=["job_id", "vocab"])
    res_data.to_csv("same_idx_work/results_processed.csv")

if __name__ == "__main__":
    main()