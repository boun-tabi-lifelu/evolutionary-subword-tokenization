from helper_classes import Sym, SymList, SymPair, MaxHeapMap
import more_itertools
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load("BLOSUM62")

# assumes the corpus is a list of strings (sequences)
# returns a list of symlists corresponding to those sequences
def corpus_to_symlist_list(corpus):
    sequences = []
    for seq in corpus:
        if len(seq) == 0: continue
        symlist = SymList()
        for sym_str in seq:
            symlist.append(Sym(sym_str))
        sequences.append(symlist)
    return sequences


# helper function, not to be used outside
def add_entry(db, sym1, sym2):
        sym1_str = sym1.literal
        sym2_str = sym2.literal
        curr_pair = db.get((sym1_str, sym2_str), None)
        if curr_pair is None:
            curr_pair = SymPair(sym1_str, sym2_str)
            db[(sym1_str, sym2_str)] = curr_pair
        curr_pair.add_pos((sym1, sym2))

# Converts a list of symlists (sequences) to a max heap of sympairs
# sorted by pair occurence frequency
def sequences_to_heap(sequences):
    pair_database = {}

    for seq in sequences:
        for sym1, sym2 in more_itertools.pairwise(seq):
            add_entry(pair_database, sym1, sym2)

    merge_heap = MaxHeapMap()
    for pair in pair_database.values():
        merge_heap.push(pair)

    return merge_heap


# Merges the given pair and adds it to the heap
# also bookkeeping the records
def merge_pair(sym_pair, merge_heap):
    # print(f"merging: {str(sym_pair)}")
    add_database = {}
    remove_database = {}
    for sym1, sym2 in sym_pair.positions:
        
        # Due to lazy removing, check if we are trying to merge an already removed pair
        if sym1.next.literal != sym_pair.right: continue
        if sym2.prev.literal != sym_pair.left: continue

        merged_sym = Sym(sym1.literal + sym2.literal)
        merged_sym.prev = sym1.prev
        merged_sym.next = sym2.next
        if sym1.prev is not None:
            pre = sym1.prev
            pre.next = merged_sym
            # print(f"Adding new entries to databases", pre, merged_sym)
            # print(f"Adding new entries to databases", pre, sym1)
            add_entry(add_database, pre, merged_sym)
            # Don't attempt to remove currently processed merge pair
            if (pre.literal, sym1.literal) != (sym_pair.left, sym_pair.right): add_entry(remove_database, pre, sym1)
        if sym2.next is not None:
            nex = sym2.next
            nex.prev = merged_sym
            # print(f"Adding new entries to databases", pre, merged_sym)
            # print(f"Adding new entries to databases", pre, sym1)
            add_entry(add_database, merged_sym, nex)
             # Don't attempt to remove currently processed merge pair
            if (sym2.literal, nex.literal) != (sym_pair.left, sym_pair.right): add_entry(remove_database, sym2, nex)
    # print(f"Sizes of the new databases add and remove: {len(add_database)}, {len(remove_database)}")
    for val in add_database.values():
        # print(f"adding: {str(val)}")
        merge_heap.push(val)
    for r_val in remove_database.values():
        # print(f"trying to remove: {str(r_val)}")
        inner_val = merge_heap.remove_by_value(r_val)
        inner_val.count -= r_val.count
        merge_heap.push(inner_val)


# Generates a sorted list of mutations (above the cutoff similarity) for the given sequence.
# Each resulting element is a tuple (m_sequence, score), where:
# - m_sequence: the mutated sequence as string
# - score: similarity score to the original sequence, a number between 0 and 1
# Results are sorted from highest to lowest score.
# First element of the results is the original sequence
# Mutations are based only on substitutions with non-negative matrix values.
def generate_mutations(seq, matrix, cutoff):
    # if len(seq) > 10:
    #     return []
    alp = matrix.alphabet
    candidates = []

    max_score = 0
    for aa in seq:
        if aa != "X":
            max_score += matrix[aa][aa]

    # Create mutation candidates for each symbol in the sequence
    for i, aa in enumerate(seq):
        candidates.append([])
        # Ignore X from calculation
        if aa == "X":
            candidates[i].append((aa, 0.0))
            continue
        # Consider substitutions with non-negative scores
        for c_aa in alp:
            score = matrix[aa][c_aa]
            if score >= -1e-4: # for floating point precision
                similarity_loss = (matrix[aa][aa] - score)/max_score
                # if the similarity loss from this particular aminoacid is large enough
                # to go under the cutoff, don't even consider it
                if similarity_loss < 1 - cutoff:
                    candidates[i].append((c_aa, similarity_loss))

    for idx, candidate in enumerate(candidates):
        candidates[idx] = sorted(candidate, key=lambda x: x[1])

    # Use dfs to search mutations that has score above tbe cutoff, fast.
    def dfs(current_seq, current_score, position):
        # Prune branches where the score is under the cutoff
        if 1-current_score < cutoff:
            return
        
        # If the string is of the required length, save it
        if position == len(seq):
            final_mutations.append((current_seq, 1-current_score))
            return
        
        # Try all candidate mutations at the current position
        for candidate, candidate_score in candidates[position]:
            # Prune if score is under the cutoff
            # This pruning works because candidate scores are sorted
            if 1-(current_score+candidate_score) < cutoff:
                break
            dfs(current_seq+candidate, current_score+candidate_score, position+1)

    final_mutations = []
    dfs("", 0, 0)
    return sorted(final_mutations, key=lambda x: -x[1])



def train_bpe(corpus = None, # A list of strings, is not necessary as long as following two parameters are given
              alphabet = None, # if not given, a corpus is expected, is not edited during execution
              sequences = None, # if not given, a corpus is expected, will be edited during execution
              merge_heap = None, # if not given, will be generated from sequences, will be edited during execution
              tokenizer_type = "default", # either "mutated" or "default"
              subs_matrix = blosum62, # Only used in tokenizer_type "mutated"
              mutation_cutoff = 0.8, # Only used in tokenizer_type "mutated"
              stop_type = "vocab_size", # either "vocab_size", "freq_cutoff" or "freq_proportion"
              stop_parameter = None # need to be set depending on the stop_type, max vocabulary size for vocab_size etc.
              ):
    try:
        if alphabet is None: 
            alphabet = []
            for seq in corpus:
                for letter in seq:
                    if letter not in alphabet:
                        alphabet.append(letter)
            alphabet.sort()
        if sequences is None:
            sequences = corpus_to_symlist_list(corpus)
        if merge_heap is None:
            merge_heap = sequences_to_heap(sequences)
    except:
        print("Did not receive proper arguments!")
    
    output_vocab = dict()
    for symbol in alphabet: # Add the alphabet in the vocabulary dictionary (is this needed?)
        output_vocab[symbol] = {
            "frequency": 0,
            "order": 0
        }

    try:
        if stop_type == "vocab_size":
            # stop upon reaching the desired vocabulary size
            stop_checker = lambda : len(output_vocab) < stop_parameter
        elif stop_type == "freq_cutoff":
            # stop when frequency of remaining pairs are less than a particular value
            stop_checker = lambda : merge_heap.peek().count > stop_parameter
        elif stop_type == "freq_proportion":
            # stop when frequency of remaining pairs are less than a proportion of the maximum frequency pair
            max_freq = merge_heap.peek().count
            cutoff = max_freq * stop_parameter
            stop_checker = lambda : merge_heap.peek().count > cutoff
        else:
            raise ValueError()
    except:
        print("Invalid value for parameter stop_type!")

    merge_counter = 0
    while stop_checker():
        merge_counter += 1

        best_pair = merge_heap.pop()
        merge_pair(best_pair, merge_heap)
        merged_string = best_pair.merged()


        # add the vocabulary entry
        if merged_string in output_vocab:
            print(f"Warning: Duplicate token {merged_string}")

        output_vocab[merged_string] = {
            "frequency": best_pair.count,
            "order": merge_counter,
            "pair": (best_pair.left, best_pair.right)
        }
        
        if tokenizer_type == "mutated":
            # Consider only the mutations with a similarity score larger than mutation cutoff
            # [1:] ignores the original string
            mutations = generate_mutations(merged_string, subs_matrix, mutation_cutoff)[1:]
            for mutated_str, score in mutations:
                pairs_to_merge = merge_heap.merged_to_pair.get(mutated_str, [])
                for pair in pairs_to_merge:
                    merge_counter += 1
                    merge_heap.remove_by_value(pair)
                    merge_pair(pair, merge_heap)

                    # add the vocabulary entry
                    if mutated_str in output_vocab:
                        print(f"Warning: Duplicate token {mutated_str}")
                    output_vocab[mutated_str] = {
                        "frequency": pair.count,
                        "order": merge_counter,
                        "pair": (pair.left, pair.right),
                        "parent": merged_string,
                        "similarity": score
                    }

    return output_vocab