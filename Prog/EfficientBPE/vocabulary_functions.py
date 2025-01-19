'''
Vocabulary entry dictionary structure

For almost all entries:
key = token text
value = {
        "frequency": Number of occurences of this merge, 
        "order": The order in which this merge is executed,
        "pair": A string tuple of the left and right token of this merge,
        "parent": Parent token string (only defined if mutated),
        "similarity": Similarity score (only defined if mutated)
}

For alphabet symbols:
key = symbol text
value = {
        "frequency": 0,
        "order": 0
}
'''
import numpy as np
import json
from tokenizers import Tokenizer

# Converts the vocab dictionary data structure into
# hugging face tokenizer data structure and returns it
def vocab_dict_to_HF_dict(vocab_dict):
    hf_skeleton = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            }
    }
    merges = []
    vocab = {}
    count = 0
    for token, data in vocab_dict.items():
        vocab[token] = count
        count += 1
        if len(token) != 1:
            merges.append(" ".join(data["pair"]))
    hf_skeleton["model"]["vocab"] = vocab
    hf_skeleton["model"]["merges"] = merges
    return hf_skeleton

# Given a vocab dictionary json as input file,
# generates a hugging face format json file in the output file
def vocab_json_to_HF_json(input_filepath, output_filepath):
    try:
        with open(input_filepath) as f:
            voc_dict = json.load(f)
        with open(output_filepath, "w") as f:
            json.dump(vocab_dict_to_HF_dict(voc_dict), f, indent=2)
    except:
        print("There was an error in either reading or writing the vocab file.")


# Given two tokenized versions of the same sequence (as two ordered token lists)
# calculate the agreement rate (of the token boundaries) between the two tokenizations
# This measure is similar to accuracy measure, calculates the rate of true positive + true negative
def calc_agreement2(tlist1, tlist2):

    # Calculate token boundary indices for list 1
    current_pos = 0
    tkn_bound_indices1 = []
    for t in tlist1:
        tkn_bound_indices1.append(current_pos)
        current_pos += len(t)
    tkn_bound_indices1.append(current_pos)
    # Calculate token boundary indices for list 2
    current_pos = 0
    tkn_bound_indices2 = []
    for t in tlist2:
        tkn_bound_indices2.append(current_pos)
        current_pos += len(t)
    tkn_bound_indices2.append(current_pos)

    if tkn_bound_indices1[-1] != tkn_bound_indices2[-1]:
        "Two token lists do not describe a sequence of same length!"

    # Convert the token boundary indices to feature vectors
    tkn_bounds1 = np.zeros((current_pos + 1), dtype='bool')
    tkn_bounds1[tkn_bound_indices1] = True
    tkn_bounds2 = np.zeros((current_pos + 1), dtype='bool')
    tkn_bounds2[tkn_bound_indices2] = True
    return np.sum(tkn_bounds1 == tkn_bounds2)/tkn_bounds1.shape[0]


# Given two tokenized versions of the same sequence (as two ordered token lists)
# calculate the agreement rate (of the token boundaries) between the two tokenizations
# calculates the Dice Sorensen coefficient
def calc_agreement(tlist1, tlist2):
    # Calculate token boundary indices for list 1
    current_pos = 0
    tkn_bound_indices1 = []
    for t in tlist1:
        tkn_bound_indices1.append(current_pos)
        current_pos += len(t)
    tkn_bound_indices1.append(current_pos)
    list1_len = current_pos
    # Calculate token boundary indices for list 2
    current_pos = 0
    tkn_bound_indices2 = []
    for t in tlist2:
        tkn_bound_indices2.append(current_pos)
        current_pos += len(t)
    tkn_bound_indices2.append(current_pos)
    list2_len = current_pos

    if tkn_bound_indices1[-1] != tkn_bound_indices2[-1]:
        "Two token lists do not describe a sequence of same length!"

    # Convert the token boundary indices to feature vectors
    max_len = list1_len if list1_len > list2_len else list2_len
    tkn_bounds1 = np.zeros((max_len + 1), dtype='int')
    tkn_bounds1[tkn_bound_indices1] = True
    tkn_bounds2 = np.zeros((max_len + 1), dtype='int')
    tkn_bounds2[tkn_bound_indices2] = True
    intersection = np.dot(tkn_bounds1, tkn_bounds2)
    return 2 * intersection / (tkn_bounds1.sum() + tkn_bounds2.sum())


def calc_dice_idx_only(index_lists):
    all_sets = [set(l) for l in index_lists]
    total_intersect = set.intersection(*all_sets)
    return len(index_lists)*len(total_intersect)/sum([len(l) for l in all_sets])


# Returns the subset of all the mutated tokens
def get_mutated(vocab):
    return {k: v for k, v in vocab.items() if "parent" in v}

# Returns the subset of all the parent tokens
def get_parents(vocab):
    return {k: v for k, v in vocab.items() if "is_parent" in v}

# Returns the set difference vocab1 \ vocab2
def set_difference(vocab1, vocab2):
    return {k: v for k,v in vocab1.items() if k not in vocab2}

# Returns the set intersection vocab1 n vocab2
def set_intersection(vocab1, vocab2):
    return {k: v for k,v in vocab1.items() if k in vocab2}


def generate_tokenizer_name(tokenizer_opts, vocab_size):
    if tokenizer_opts['is_mut']:
        tokenizer_name = f"mutBPE{' pre' if tokenizer_opts['is_pretokenizer'] else ''} {tokenizer_opts['subs_matrix']} {tokenizer_opts['mutation_cutoff']} {tokenizer_opts['min_mutation_freq']} {vocab_size}"
    else:
        tokenizer_name = f"stdBPE{' pre' if tokenizer_opts['is_pretokenizer'] else ''} {vocab_size}"
    return tokenizer_name

def generate_tokenizer_filename(tokenizer_opts, vocab_size):
    if tokenizer_opts['is_mut']:
        file_name = f"{tokenizer_opts['dataset']}{'pre' if tokenizer_opts['is_pretokenizer'] else ''}_mutbpe_{tokenizer_opts['mutation_cutoff']}_{tokenizer_opts['min_mutation_len']}"
        file_name += f"_{tokenizer_opts['max_mutation_len']}_{tokenizer_opts['min_mutation_freq']}_{vocab_size}"
    else:
        file_name = f"{tokenizer_opts['dataset']}{'pre' if tokenizer_opts['is_pretokenizer'] else ''}_bpe_{vocab_size}"
    return file_name


def load_tokenizer(tokenizer_opts, folder_path = "/cta/share/users/mutbpe/tokenizers", hf_or_vocab = 'hf'):
    tokenizer_list = {}
    if tokenizer_opts['is_mut']:
        for vocab_size in tokenizer_opts['vocab_size']:
            tokenizer_name = generate_tokenizer_name(tokenizer_opts, vocab_size)
            file_name = generate_tokenizer_filename(tokenizer_opts, vocab_size)
            if hf_or_vocab == 'hf':
                file_path = f"{folder_path}/{tokenizer_opts['subs_matrix']}/hf_{file_name}"
                tokenizer_list[tokenizer_name] = Tokenizer.from_file(f"{file_path}.json")
            else:
                file_path = f"{folder_path}/{tokenizer_opts['subs_matrix']}/{file_name}"
                with open(f"{file_path}.json") as json_file:
                    tokenizer_list[tokenizer_name] = json.load(json_file)
                
    else:
        for vocab_size in tokenizer_opts['vocab_size']:
            tokenizer_name = generate_tokenizer_name(tokenizer_opts, vocab_size)
            file_name = generate_tokenizer_filename(tokenizer_opts, vocab_size)
            if hf_or_vocab == 'hf':
                file_path = f"{folder_path}/{'blosum62'}/hf_{file_name}"
                tokenizer_list[tokenizer_name] = Tokenizer.from_file(f"{file_path}.json")
            else:
                file_path = f"{folder_path}/{'blosum62'}/{file_name}"
                with open(f"{file_path}.json") as json_file:
                    tokenizer_list[tokenizer_name] = json.load(json_file)

    return tokenizer_list

def load_tokenizers(tokenizer_opts_list, hf_or_vocab = 'hf'):
    """
    # 'dataset': {'uniref50', 'uniref90'}
    # 'is_pretokenizer': {True, False}
    # 'subs_matrix': {'blosum45', 'blosum62', 'pam70', 'pam250'}
    # 'mutation_cutoff': {0.7, 0.8, 0.9}
    # 'min_mutation_freq': {0, 0.05,. 0.005}
    # 'min_mutation_len': {3}
    # 'max_mutation_len': {12}
    # 'vocab_size': list=[800, 1600, 3200, 6400, 12800, 25600, 51200]

    vocab_sizes = [800, 3200, 12800]
    uniref_id = "50"

    tokenizer_opts_list = [
        {
            'is_mut': False,
            'dataset': f'uniref{uniref_id}',
            'is_pretokenizer': False,
            'vocab_size': vocab_sizes
        },
        {
            'is_mut': True,
            'dataset': f'uniref{uniref_id}',
            'is_pretokenizer': True,
            'subs_matrix': 'blosum62',
            'mutation_cutoff': 0.7,
            'min_mutation_freq': 0.05,
            'min_mutation_len': 3,
            'max_mutation_len': 12,
            'vocab_size': vocab_sizes
        }
    ]

    tokenizer_list = load_tokenizers(tokenizer_opts_list, 'hf')
    inner_vocab_list = load_tokenizers(tokenizer_opts_list, 'vocab')
    """
    tokenizer_list = {}
    for tokenizer_opts in tokenizer_opts_list:
        tokenizer_list.update(load_tokenizer(tokenizer_opts, hf_or_vocab=hf_or_vocab))
    return tokenizer_list