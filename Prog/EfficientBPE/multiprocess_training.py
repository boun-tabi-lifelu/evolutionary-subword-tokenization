import multiprocessing, sqlite3, json
import pandas as pd
import bpe_functions, vocabulary_functions

def opts_to_file_name(prefix, opts):
    if opts["tokenizer_type"] == "default":
        return f"{prefix}_bpe_{opts["stop_parameter"]}.json"
    elif opts["tokenizer_type"] == "mutated":
        return f"{prefix}_mutbpe_{opts["mutation_cutoff"]}_{opts["min_mutation_len"]}_{opts["max_mutation_len"]}_{opts["min_mutation_freq"]}_{opts["stop_parameter"]}.json"
    return None

def training(args):
    options, save_filepath, hf_filepath = args
    print(f"Training started for {save_filepath}")
    vocab = bpe_functions.train_bpe(**options)
    print(f"Training finished for {save_filepath}")
    with open(save_filepath, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Generating hugging face format for {save_filepath}")
    vocabulary_functions.vocab_json_to_HF_json(save_filepath, hf_filepath)

    



if __name__ == "__main__":
    db_file = "/cta/share/users/uniprot/human/human.db"
    conn = sqlite3.connect(db_file)
    df_uniprot_human_seqs = pd.read_sql(f"SELECT Sequence FROM proteins WHERE Entry IN (SELECT uniprot_accession FROM uniref50_distilled)", conn)
    conn.close()

    filtered_sequences = df_uniprot_human_seqs[
        (df_uniprot_human_seqs["Sequence"].str.count("X") <= 1) &
        (df_uniprot_human_seqs["Sequence"].str.count("B") <= 1) &
        (df_uniprot_human_seqs["Sequence"].str.count("U") <= 1) &
        (df_uniprot_human_seqs["Sequence"].str.count("Z") <= 1)
    ]["Sequence"].tolist()

    alphabet = ['A', 'R', 'N', 'D', 'C', 'E', 
                'Q', 'G', 'H', 'I', 'L', 'K', 
                'M', 'F', 'P', 'S', 'T', 'W', 
                'Y', 'V', 'U', 'O', 'X', 'B', 
                'Z', 'J']
    corpus = filtered_sequences
    vocab_size = 51200
    save_folder = "/cta/share/users/mutbpe/tokenizers/"
    argument_set_cutoff = [0.7, 0.8, 0.9]
    argument_set_mutfreq = [0, 0.005, 0.05]

    arguments = [
        {
        "corpus": corpus,
        "alphabet": alphabet,
        "tokenizer_type": "default",
        "stop_type": "vocab_size",
        "stop_parameter": vocab_size
        }
    ]
    for cutoff in argument_set_cutoff:
        for mutfreq in argument_set_mutfreq:
            arguments.append(
                {
                    "corpus": corpus,
                    "alphabet": alphabet,
                    "tokenizer_type": "mutated",
                    "mutation_cutoff": cutoff,
                    "min_mutation_len": 3,
                    "max_mutation_len": 12,
                    "min_mutation_freq": mutfreq,
                    "stop_type": "vocab_size",
                    "stop_parameter": vocab_size
                        
                }
            )
    save_folders = [save_folder + opts_to_file_name("uniref50", opts) for opts in arguments]
    hf_folders = [save_folder + "hf_" + opts_to_file_name("uniref50", opts) for opts in arguments]
    # print(save_folders)
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(training, zip(arguments, save_folders, hf_folders))
    