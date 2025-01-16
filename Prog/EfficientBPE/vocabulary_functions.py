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
    tkn_bounds1 = np.zeros((current_pos + 1), dtype='int')
    tkn_bounds1[tkn_bound_indices1] = True
    tkn_bounds2 = np.zeros((current_pos + 1), dtype='int')
    tkn_bounds2[tkn_bound_indices2] = True
    intersection = np.dot(tkn_bounds1, tkn_bounds2)
    return 2 * intersection / (tkn_bounds1.sum() + tkn_bounds2.sum())

# Assumes the indices are 0-indexed. 
def calc_dice_idx_only(indices1, indices2):
    seq_length = max(max(indices1), max(indices2))
    bounds1 = np.zeros(seq_length + 1, dtype="int")
    bounds2 = np.zeros(seq_length + 1, dtype="int")
    bounds1[indices1] = 1
    bounds2[indices2] = 1
    intersection = np.dot(bounds1, bounds2)
    return 2 * intersection / (len(indices1) + len(indices2)) 


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