import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import EsmTokenizer, EsmForMaskedLM
import torch
import torch.nn.functional as F
from Bio.Align import substitution_matrices
import ast
import random

# --- Pre-computation Function ---

def precompute_alternatives(sub_matrix):
    """
    Pre-computes a lookup table for amino acid substitutions that have scores equal to or 
    greater than a target mutation.

    This optimization speeds up the search for 'Alternative' sequences in the PUMA experiment.
    For every possible amino acid mutation (A -> B), it identifies all other amino acids (C)
    where Score(A, C) >= Score(A, B).

    Parameters
    ----------
    sub_matrix : Bio.Align.substitution_matrices.Array
        A substitution matrix (e.g., BLOSUM62, PAM70) loaded via Biopython.

    Returns
    -------
    precomputed_alternatives : dict
        A nested dictionary structure: 
        {original_aa: {mutated_aa: [(alternative_aa, score_diff), ...], ...}, ...}
        where `score_diff` is (Score(orig, alt) - Score(orig, mut)).
    """
    print("Pre-computing alternative amino acid scores...")
    AMINO_ACIDS = 'ABCDEFGHIKLMNPQRSTVWYZ'
    precomputed_alternatives = {orig_aa: {} for orig_aa in AMINO_ACIDS}

    for orig_aa in tqdm(AMINO_ACIDS):
        for mut_aa in AMINO_ACIDS:
            if orig_aa == mut_aa:
                continue
            
            try:
                score_mutation = sub_matrix[(orig_aa, mut_aa)]
            except KeyError:
                continue

            possible_alternatives = []
            for alt_aa in AMINO_ACIDS:
                if alt_aa in (orig_aa, mut_aa):
                    continue
                try:
                    score_alternative = sub_matrix[(orig_aa, alt_aa)]
                    if score_alternative >= score_mutation:
                        possible_alternatives.append((alt_aa, score_alternative-score_mutation))
                except KeyError:
                    continue
            
            # Sort by score (descending) and store only the amino acid
            possible_alternatives.sort(key=lambda x: x[1], reverse=True)
            # precomputed_alternatives[orig_aa][mut_aa] = [alt[0] for alt in possible_alternatives]
            precomputed_alternatives[orig_aa][mut_aa] = possible_alternatives.copy()
            
    return precomputed_alternatives


# --- Enhanced Helper Functions ---

def find_alternative_replacement_optimized(
    original_token, 
    mutation_replacement, 
    replacement_pool_set, 
    precomputed_alts,
    vocab_lineage
):
    """
    Finds a valid 'Alternative' token for the control experiment using pre-computed scores.

    The alternative token must satisfy specific PUMA constraints:
    1. Must be in the PUMA vocabulary.
    2. Must satisfy the score constraint: Score(Original, Alternative) >= Score(Original, Mutation).
    3. Must not be a direct sibling/child of the original token (checked via `replacement_pool_set`).
    4. Must have been added to the vocabulary *later* than the mutation to preclude close genealogical ties (`vocab_lineage` order check).

    Parameters
    ----------
    original_token : str
        The protein unit (token) from the original sequence.
    mutation_replacement : str
        The PUMA-defined sibling/child token chosen as the primary mutation.
    replacement_pool_set : set
        A set containing the original token and its known family members (siblings/children)
        to ensure the alternative is not closely related.
    precomputed_alts : dict
        The lookup table returned by `precompute_alternatives`.
    vocab_lineage : dict
        Metadata for the vocabulary, including 'order' (insertion index) for each token.

    Returns
    -------
    token : str
        The selected alternative token. Returns `mutation_replacement` (original mutation)
        if no valid alternative is found.
    score : float
        The substitution score of the found alternative. Returns 0 if none found.
    """
    if len(original_token) != len(mutation_replacement):
        return mutation_replacement, 0

    diff_positions = [
        i for i, (orig_aa, mut_aa) in enumerate(zip(original_token, mutation_replacement)) 
        if orig_aa != mut_aa
    ]

    if not diff_positions:
        return mutation_replacement, 0

    # Get mutation replacement order for comparison
    mutation_order = vocab_lineage.get(mutation_replacement, {}).get('order', 0)
    
    # Generate all possible alternative tokens
    alternative_candidates = []
    
    def generate_alternatives(token_list, pos_index, current_score):
        if pos_index >= len(diff_positions):
            # Complete alternative generated
            new_token = "".join(token_list)
            alternative_candidates.append((new_token, current_score))
            return
        
        pos = diff_positions[pos_index]
        original_aa = original_token[pos]
        mutated_aa = mutation_replacement[pos]
        
        # Get alternatives for this position from precomputed table
        alternatives_with_scores = precomputed_alts.get(original_aa, {}).get(mutated_aa, [])
        
        for alt_aa, score in alternatives_with_scores:
            new_token_list = token_list.copy()
            new_token_list[pos] = alt_aa
            generate_alternatives(new_token_list, pos_index + 1, current_score + score)
    
    # Start generation with mutation_replacement as base
    start_token_list = list(mutation_replacement)
    generate_alternatives(start_token_list, 0, 0)
    
    alternative_candidates = [(new_token, score, vocab_lineage[new_token].get('order', len(vocab_lineage[new_token]))) for new_token, score in alternative_candidates if new_token in vocab_lineage]

    # Sort by score in descending order
    # alternative_candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    alternative_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Find the highest scoring alternative that meets constraints
    for new_token, score, _ in alternative_candidates:
        if (new_token not in replacement_pool_set) and (new_token in vocab_lineage) and (vocab_lineage[new_token].get('order', 0) > mutation_order):
            return new_token, score
    
    # If no valid alternative found, return original mutation
    return mutation_replacement, 0


def create_random_alternative_baseline(original_token, mutation_replacement, alternative_replacement, sub_matrix):
    """
    Creates a random baseline token for control purposes.
    
    This replaces the differing amino acid positions with a random amino acid 
    (excluding the original) to establish a random baseline for ESM-2 preference.

    Parameters
    ----------
    original_token : str
        The original protein unit.
    mutation_replacement : str
        The PUMA-defined mutation unit.
    alternative_replacement : str
        The selected high-scoring alternative unit.
    sub_matrix : Bio.Align.substitution_matrices.Array
        Substitution matrix (used in logic if extensions are needed, currently strictly random).

    Returns
    -------
    str
        A token string with random amino acid substitutions at the differing positions.
    """
    if len(original_token) != len(mutation_replacement):
        return mutation_replacement
    
    AMINO_ACIDS = list('ABCDEFGHIKLMNPQRSTVWYZ')
    alternative_token = list(mutation_replacement)
    
    for i, (orig_aa, mut_aa, alt_aa) in enumerate(zip(original_token, mutation_replacement, alternative_replacement)):
        if orig_aa != mut_aa and mut_aa != alt_aa:
            alternative_token[i] = random.choice([aa for aa in AMINO_ACIDS if aa != orig_aa])
    
    return ''.join(alternative_token)


# --- Enhanced Main Function ---

def run_mutation_experiment_optimized(df_protein, tokenizer_list, vocab_lineage_list, sub_matrix_precomputed_alternatives, create_baseline=True):
    """
    Generates the experimental dataset containing Mutated, Alternative, and Baseline sequences.

    For every token in a protein sequence that belongs to a PUMA family (has a parent/siblings),
    this function generates:
    1. A 'Mutated' sequence (using a PUMA sibling).
    2. An 'Alternative' sequence (using a non-family, high-scoring token).
    3. A 'Baseline' sequence (random substitution).

    Parameters
    ----------
    df_protein : pd.DataFrame
        Input dataframe containing protein sequences tokenized by various PUMA models.
    tokenizer_list : dict
        Dictionary of tokenizer objects keyed by model name.
    vocab_lineage_list : dict
        Dictionary of lineage metadata for each tokenizer.
    sub_matrix_precomputed_alternatives : dict
        Precomputed substitution scores for different matrices (BLOSUM62, PAM70, etc.).
    create_baseline : bool, default=True
        Whether to generate random baseline sequences.

    Returns
    -------
    df_results : pd.DataFrame
        The input dataframe enriched with columns for 'mutated', 'alternative', and 'baseline' token sequences.
    change_counters : dict
        Count of successful substitutions performed per model.
    change_scores : dict
        Cumulative substitution scores for alternatives per model.
    """
    np.random.seed(42)
    df_results = df_protein.copy()
    token_cols = list(tokenizer_list.keys())
    change_counters = {}
    change_scores = {}

    for col_name in token_cols:
        print(f"\nProcessing column: {col_name}")
        change_counters[col_name] = 0
        change_scores[col_name] = 0

        sub_matrix_name = col_name.split()[2] if col_name.split()[1] == 'pre' else col_name.split()[1]
        precomputed_alternatives = sub_matrix_precomputed_alternatives[sub_matrix_name]
        sub_matrix = substitution_matrices.load(sub_matrix_name.upper())
        
        all_mutated_sequences = []
        all_alternative_sequences = []
        all_baseline_sequences = [] if create_baseline else None
        
        vocab_lineage = vocab_lineage_list.get(col_name, {})

        for tokenized_sequence in tqdm(df_results[col_name]):
            new_mutated_sequence = []
            new_alternative_sequence = []
            new_baseline_sequence = [] if create_baseline else None
            
            for token in tokenized_sequence:
                token_info = vocab_lineage.get(token)
                
                if not token_info:
                    new_mutated_sequence.append(token)
                    new_alternative_sequence.append(token)
                    if create_baseline:
                        new_baseline_sequence.append(token)
                    continue
                
                if 'parent_mutation' in token_info and token_info['parent_mutation']:
                    parent_mutation_token = token_info['parent_mutation']
                    replacement_pool = vocab_lineage[parent_mutation_token].get('child_mutation', [])
                    replacement_pool_set = set(replacement_pool + [token, parent_mutation_token])
                else:
                    new_mutated_sequence.append(token)
                    new_alternative_sequence.append(token)
                    if create_baseline:
                        new_baseline_sequence.append(token)
                    continue

                for mutation_replacement in replacement_pool:
                    alternative_replacement, alt_score = find_alternative_replacement_optimized(
                        token, 
                        mutation_replacement, 
                        replacement_pool_set, 
                        precomputed_alternatives,
                        vocab_lineage
                    )
                    
                    if alternative_replacement != mutation_replacement:
                        new_mutated_sequence.append(mutation_replacement)
                        new_alternative_sequence.append(alternative_replacement)
                        if create_baseline:
                            baseline_replacement = create_random_alternative_baseline(
                                token, mutation_replacement, alternative_replacement, sub_matrix
                            )
                            new_baseline_sequence.append(baseline_replacement)
                        change_counters[col_name] += 1
                        change_scores[col_name] += alt_score
                        break
                else:
                    new_mutated_sequence.append(token)
                    new_alternative_sequence.append(token)
                    if create_baseline:
                        new_baseline_sequence.append(token)

            all_mutated_sequences.append(new_mutated_sequence)
            all_alternative_sequences.append(new_alternative_sequence)
            if create_baseline:
                all_baseline_sequences.append(new_baseline_sequence)

        df_results[f'{col_name} mutated'] = all_mutated_sequences
        df_results[f'{col_name} alternative'] = all_alternative_sequences
        if create_baseline:
            df_results[f'{col_name} baseline'] = all_baseline_sequences
        print(f"Changes: {change_counters[col_name]}, Scores: {change_scores[col_name]}, Score perc: {change_scores[col_name] / change_counters[col_name] if change_counters[col_name] > 0 else 0}")
        
    return df_results, change_counters, change_scores


# --- Enhanced ESM-2 Functions ---

# facebook/model_name
# Checkpoint name       Num layers  Num parameters  Dim count
# esm2_t48_15B_UR50D    48          15B             5120       
# esm2_t36_3B_UR50D     36          3B              2560        
# esm2_t33_650M_UR50D   33          650M            1280     
# esm2_t30_150M_UR50D   30          150M            640      
# esm2_t12_35M_UR50D    12          35M             480      
# esm2_t6_8M_UR50D      6           8M              320      

def get_esm2_model_and_tokenizer(model_name="facebook/esm2_t33_650M_UR50D"):
    """
    Initializes the ESM-2 model and tokenizer for masked language modeling.

    Parameters
    ----------
    model_name : str, default="facebook/esm2_t33_650M_UR50D"
        Hugging Face model identifier.

    Returns
    -------
    model : EsmForMaskedLM
        The loaded model on the appropriate device (CUDA/CPU).
    tokenizer : EsmTokenizer
        The associated tokenizer.
    device : torch.device
        The device being used.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    
    print(f"Loaded ESM-2 model: {model_name}")
    return model, tokenizer, device


def get_esm2_embeddings_and_logits_batch(sequences, model, tokenizer, device, batch_size=32, max_length=1024, extract_layers=None):
    """
    Computes ESM-2 embeddings and full-sequence logits in batches.

    Parameters
    ----------
    sequences : list of str
        List of protein sequences.
    model : EsmForMaskedLM
        Loaded ESM-2 model.
    tokenizer : EsmTokenizer
        Loaded ESM tokenizer.
    device : torch.device
        Computing device.
    batch_size : int, default=32
        Batch size for inference.
    max_length : int, default=1024
        Truncation length for sequences.
    extract_layers : list of int, optional
        List of layer indices to extract embeddings from. Defaults to [-1] (last layer).

    Returns
    -------
    sequence_data : dict
        Dictionary mapping input sequence to a dict containing 'logits' and 'embeddings'.
    """
    if extract_layers is None:
        extract_layers = [-1]  # Last layer only
    
    sequence_data = {}
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing embedding batches"):
        batch_sequences = sequences[i:i + batch_size]
        
        inputs = tokenizer(
            batch_sequences, 
            return_tensors="pt", 
            padding='longest', 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            batch_logits = outputs.logits
            hidden_states = outputs.hidden_states
        
        for j, seq in enumerate(batch_sequences):
            seq_len = len(seq)
            
            # Store logits (for probability analysis)
            logits = batch_logits[j, 1:seq_len+1].cpu()  # Remove <cls>, keep actual sequence
            
            # Store embeddings from specified layers
            embeddings = {}
            for layer_idx in extract_layers:
                layer_embeddings = hidden_states[layer_idx][j, 1:seq_len+1].cpu()
                embeddings[layer_idx] = layer_embeddings
            
            sequence_data[seq] = {
                'logits': logits,
                'embeddings': embeddings
            }
    
    return sequence_data

def get_masked_logits_batch(sequence_position_pairs, model, tokenizer, device, batch_size=32, max_length=1024):
    """
    Computes logits for specific masked positions in sequences.

    This function masks the specified position in a sequence (e.g., A[MASK]C) 
    and returns the model's logits for that position, allowing probability comparison 
    of different amino acids at that specific context.

    Parameters
    ----------
    sequence_position_pairs : list of tuple
        List of (sequence, position_index) tuples.
    model : EsmForMaskedLM
        Loaded ESM-2 model.
    tokenizer : EsmTokenizer
        Loaded ESM tokenizer.
    device : torch.device
        Computing device.
    batch_size : int, default=32
        Batch size.
    max_length : int, default=1024
        Max sequence length.

    Returns
    -------
    masked_logits : dict
        Dictionary mapping (sequence, position) -> torch.Tensor (logits for the masked position).
    """
    masked_logits = {}
    
    # Create all masked sequences
    masked_sequences = []
    sequence_position_map = {}
    
    for seq, pos in sequence_position_pairs:
        if pos >= len(seq):
            continue
        
        seq_list = list(seq)
        seq_list[pos] = tokenizer.mask_token
        masked_seq = ''.join(seq_list)
        
        masked_sequences.append(masked_seq)
        sequence_position_map[len(masked_sequences) - 1] = (seq, pos)
    
    # Process masked sequences in batches
    for i in tqdm(range(0, len(masked_sequences), batch_size), desc="Processing masked sequences"):
        batch_sequences = masked_sequences[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_sequences, 
            return_tensors="pt", 
            padding='longest', 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_logits = outputs.logits
        
        # Extract logits at masked positions
        for j, masked_seq in enumerate(batch_sequences):
            batch_idx = i + j
            if batch_idx not in sequence_position_map:
                continue
                
            original_seq, position = sequence_position_map[batch_idx]
            
            # Find mask token position in tokenized sequence
            mask_token_indices = (inputs.input_ids[j] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_token_indices) > 0:
                mask_pos = mask_token_indices[0]
                position_logits = batch_logits[j, mask_pos, :].cpu()
                masked_logits[(original_seq, position)] = position_logits
    
    return masked_logits


# --- Analysis Functions ---

def reconstruct_sequence_from_tokens(tokens):
    """
    Reconstructs an amino acid sequence string from a list of tokens.
    
    Parameters
    ----------
    tokens : list or str
        List of tokens or a string representation of a list.
        
    Returns
    -------
    str
        The concatenated protein sequence.
    """
    if isinstance(tokens, str):
        tokens = ast.literal_eval(tokens)
    return ''.join(tokens)


def find_differing_amino_acid_positions(original_seq, mutated_seq, alternative_seq, baseline_seq=None):
    """
    Identifies indices where amino acids differ across the original and variant sequences.
    
    Parameters
    ----------
    original_seq : str
        Reference protein sequence.
    mutated_seq : str
        Sequence with PUMA mutation.
    alternative_seq : str
        Sequence with alternative substitution.
    baseline_seq : str, optional
        Sequence with random baseline substitution.
    
    Returns
    -------
    differing_positions : list of int
        List of 0-based indices where the sequences differ.
    """
    differing_positions = []
    sequences = [mutated_seq, alternative_seq]
    if baseline_seq:
        sequences.append(baseline_seq)
    
    min_len = min(len(original_seq), *[len(seq) for seq in sequences])
    
    for i in range(min_len):
        original_aa = original_seq[i]
        # differs = any(seq[i] != original_aa for seq in sequences)
        differs = all(seq[i] != original_aa for seq in sequences)
        if differs:
            differing_positions.append(i)
    
    return differing_positions


def compute_rank_based_metrics_optimized(original_seq, mutated_seq, alternative_seq, baseline_seq, differing_positions, masked_logits_dict, tokenizer):
    """
    Calculates win/loss metrics by comparing ESM-2 probabilities of different amino acids.

    For each differing position, it retrieves the masked logits ($L(\cdot|[MASK])$) and 
    compares the probabilities of the Original, Mutated (Sibling), Alternative, and Baseline residues.

    Parameters
    ----------
    original_seq : str
        The original protein sequence (used as key for logits).
    mutated_seq : str
        Sequence containing the PUMA sibling mutation.
    alternative_seq : str
        Sequence containing the alternative substitution.
    baseline_seq : str
        Sequence containing the random baseline substitution.
    differing_positions : list of int
        Indices to analyze.
    masked_logits_dict : dict
        Precomputed logits from `get_masked_logits_batch`.
    tokenizer : EsmTokenizer
        Tokenizer to convert amino acids to IDs.

    Returns
    -------
    rank_metrics : dict
        Dictionary containing counts of pairwise 'wins' (e.g., 'mut_wins_vs_alt').
    """
    rank_metrics = {'orig_wins_vs_mut': 0, 'orig_wins_vs_alt': 0, 'orig_wins_vs_base': 0, 'mut_wins_vs_alt': 0, 'mut_wins_vs_base': 0, 'alt_wins_vs_base': 0, 'total_positions': 0}
    
    if not differing_positions:
        return rank_metrics
    
    for pos in differing_positions:
        if pos >= len(original_seq):
            continue
            
        # Get precomputed masked logits
        masked_logits = masked_logits_dict.get((original_seq, pos))
        if masked_logits is None:
            continue
        
        # Convert to probabilities
        position_probs = F.softmax(masked_logits, dim=-1)
        
        # Get amino acids at this position
        orig_aa = original_seq[pos]
        mut_aa = mutated_seq[pos] if pos < len(mutated_seq) else orig_aa
        alt_aa = alternative_seq[pos] if pos < len(alternative_seq) else orig_aa
        base_aa = baseline_seq[pos] if baseline_seq and pos < len(baseline_seq) else orig_aa
        
        # Get token IDs for amino acids
        orig_token_id = tokenizer.convert_tokens_to_ids(orig_aa)
        mut_token_id = tokenizer.convert_tokens_to_ids(mut_aa)
        alt_token_id = tokenizer.convert_tokens_to_ids(alt_aa)
        base_token_id = tokenizer.convert_tokens_to_ids(base_aa) if baseline_seq else None
        
        # Get probabilities
        if all(tid is not None for tid in [orig_token_id, mut_token_id, alt_token_id]):
            orig_prob = position_probs[orig_token_id].item()
            mut_prob = position_probs[mut_token_id].item()
            alt_prob = position_probs[alt_token_id].item()
            
            # Compare original vs others
            if orig_prob > mut_prob:
                rank_metrics['orig_wins_vs_mut'] += 1
            if orig_prob > alt_prob:
                rank_metrics['orig_wins_vs_alt'] += 1
            
            # Compare mutation vs alternative
            if mut_prob > alt_prob:
                rank_metrics['mut_wins_vs_alt'] += 1
            
            # Compare with baseline if available
            if base_token_id is not None:
                base_prob = position_probs[base_token_id].item()
                if orig_prob > base_prob:
                    rank_metrics['orig_wins_vs_base'] += 1
                if mut_prob > base_prob:
                    rank_metrics['mut_wins_vs_base'] += 1
                if alt_prob > base_prob:
                    rank_metrics['alt_wins_vs_base'] += 1
            
            rank_metrics['total_positions'] += 1
    
    return rank_metrics


def run_enhanced_protein_similarity_experiment(
    df_protein_oma,
    tokenizer_list,
    batch_size=32, 
    max_length=514, 
    include_rank_analysis=True,
    include_baseline=True
):
    """
    Main execution wrapper for the ESM-2 similarity and likelihood experiment.

    Steps:
    1. Loads ESM-2 model.
    2. Collects all unique sequences and identifies positions requiring masking.
    3. Computes masked logits in batches (efficient GPU usage).
    4. Iterates through proteins and tokenizers to calculate pairwise win rates 
       (Mutation vs Alternative, Mutation vs Baseline, etc.).

    Parameters
    ----------
    df_protein_oma : pd.DataFrame
        Dataframe containing sequences and tokenized variants.
    tokenizer_list : dict
        Dict of PUMA tokenizers.
    batch_size : int, default=32
        Inference batch size.
    max_length : int, default=514
        Max sequence length for ESM-2.
    include_rank_analysis : bool, default=True
        Whether to perform the logit ranking analysis.
    include_baseline : bool, default=True
        Whether to include comparisons against the random baseline.

    Returns
    -------
    pd.DataFrame
        Summary results containing win rates and statistics for each tokenizer model.
    """
    
    # Initialize ESM-2 model
    model, tokenizer, device = get_esm2_model_and_tokenizer()
    
    # Identify tokenizer columns
    tokenizer_columns = list(tokenizer_list.keys())
    print(f"Found {len(tokenizer_columns)} tokenizer columns to process")
    
    # Collect all unique sequences and positions that need masking
    print("Collecting unique sequences and masking positions...")
    unique_sequences = set()
    sequence_position_pairs = set()
    
    for tokenizer_col in tokenizer_columns:
        mutated_col = tokenizer_col + ' mutated'
        alternative_col = tokenizer_col + ' alternative'
        baseline_col = tokenizer_col + ' baseline' if include_baseline else None
        
        for idx, row in df_protein_oma.iterrows():
            try:
                original_tokens = row[tokenizer_col]
                mutated_tokens = row[mutated_col]
                alternative_tokens = row[alternative_col]
                
                original_seq = reconstruct_sequence_from_tokens(original_tokens)
                mutated_seq = reconstruct_sequence_from_tokens(mutated_tokens)
                alternative_seq = reconstruct_sequence_from_tokens(alternative_tokens)
                
                unique_sequences.update([original_seq, mutated_seq, alternative_seq])
                
                if include_baseline and baseline_col in row:
                    baseline_tokens = row[baseline_col]
                    baseline_seq = reconstruct_sequence_from_tokens(baseline_tokens)
                    unique_sequences.add(baseline_seq)
                
                # Collect positions that need masking for rank analysis
                if include_rank_analysis:
                    baseline_seq_for_diff = reconstruct_sequence_from_tokens(row[baseline_col]) if include_baseline and baseline_col in row else None
                    differing_positions = find_differing_amino_acid_positions(
                        original_seq, mutated_seq, alternative_seq, baseline_seq_for_diff
                    )
                    
                    # Add all differing positions for this sequence to masking list
                    for pos in differing_positions:
                            sequence_position_pairs.add((original_seq, pos))
                        
            except Exception as e:
                continue
    
    unique_sequences = list(unique_sequences)
    sequence_position_pairs = list(sequence_position_pairs)
    print(f"Found {len(unique_sequences)} unique sequences")
    print(f"Found {len(sequence_position_pairs)} positions to mask")

    sequence_data = {}
    
    # Precompute masked logits for all positions
    masked_logits_dict = {}
    if include_rank_analysis:
        print("Precomputing masked logits for all positions...")
        masked_logits_dict = get_masked_logits_batch(
            sequence_position_pairs, model, tokenizer, device, batch_size, max_length
        )
    
    # Clear GPU memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Process each tokenizer
    all_results = []
    
    for tokenizer_col in tokenizer_columns:
        print(f"\nProcessing tokenizer: {tokenizer_col}")
        
        # Get substitution matrix for BLOSUM analysis
        sub_matrix_name = tokenizer_col.split()[2] if tokenizer_col.split()[1] == 'pre' else tokenizer_col.split()[1]
        sub_matrix = substitution_matrices.load(sub_matrix_name.upper())
        
        mutated_col = tokenizer_col + ' mutated'
        alternative_col = tokenizer_col + ' alternative'
        baseline_col = tokenizer_col + ' baseline' if include_baseline else None
        
        # Initialize result containers
        results = {
            'rank_metrics': [],
            'position_counts': []
        }
        
        processed_proteins = 0
        
        for idx, row in tqdm(df_protein_oma.iterrows(), total=len(df_protein_oma), desc="Processing proteins"):
            try:
                # Reconstruct sequences
                original_seq = reconstruct_sequence_from_tokens(row[tokenizer_col])
                mutated_seq = reconstruct_sequence_from_tokens(row[mutated_col])
                alternative_seq = reconstruct_sequence_from_tokens(row[alternative_col])
                baseline_seq = None
                
                if include_baseline and baseline_col in row:
                    baseline_seq = reconstruct_sequence_from_tokens(row[baseline_col])
                
                # Skip if no changes
                if (original_seq == mutated_seq and original_seq == alternative_seq and 
                    (not include_baseline or original_seq == baseline_seq)):
                    continue
                
                # Find differing positions
                differing_positions = find_differing_amino_acid_positions(
                    original_seq, mutated_seq, alternative_seq, baseline_seq
                )
                
                if not differing_positions:
                    continue
                
                results['position_counts'].append(len(differing_positions))
                    
                # Optimized rank analysis
                if include_rank_analysis:
                    rank_metrics = compute_rank_based_metrics_optimized(
                        original_seq, mutated_seq, alternative_seq, baseline_seq, differing_positions, 
                        masked_logits_dict, tokenizer
                    )
                    results['rank_metrics'].append(rank_metrics)
                
                processed_proteins += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Compile results for this tokenizer (same as before)
        tokenizer_results = {
            'tokenizer': tokenizer_col,
            'processed_proteins': processed_proteins,
            'total_differing_positions': sum(results['position_counts']),
            'avg_differing_positions': np.mean(results['position_counts']) if results['position_counts'] else 0
        }
        
        # Add rank results with new metrics
        if results['rank_metrics']:
            total_wins_mut = sum(r['orig_wins_vs_mut'] for r in results['rank_metrics'])
            total_wins_alt = sum(r['orig_wins_vs_alt'] for r in results['rank_metrics'])
            total_wins_base = sum(r['orig_wins_vs_base'] for r in results['rank_metrics'])
            total_wins_mutalt = sum(r['mut_wins_vs_alt'] for r in results['rank_metrics'])
            total_wins_mutbase = sum(r['mut_wins_vs_base'] for r in results['rank_metrics'])
            total_wins_altbase = sum(r['alt_wins_vs_base'] for r in results['rank_metrics'])
            total_positions = sum(r['total_positions'] for r in results['rank_metrics'])
            
            if total_positions > 0:
                tokenizer_results['rank_win_rate_vs_mut'] = total_wins_mut / total_positions
                tokenizer_results['rank_win_rate_vs_alt'] = total_wins_alt / total_positions
                tokenizer_results['rank_win_rate_vs_base'] = total_wins_base / total_positions
                tokenizer_results['rank_win_rate_mut_vs_alt'] = total_wins_mutalt / total_positions
                tokenizer_results['rank_win_rate_mut_vs_base'] = total_wins_mutbase / total_positions
                tokenizer_results['rank_win_rate_alt_vs_base'] = total_wins_altbase / total_positions
        
        all_results.append(tokenizer_results)
    
    return pd.DataFrame(all_results)