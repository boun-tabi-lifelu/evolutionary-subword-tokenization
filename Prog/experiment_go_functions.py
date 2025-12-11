import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import random
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import spearmanr
from scipy.sparse import spmatrix
import scipy.sparse as sp

from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction 

from transformers import EsmTokenizer, EsmForMaskedLM
import torch

from typing import List, Dict, Tuple, Set, Any, Optional, Union


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
    Initialize and load the ESM-2 model and tokenizer from the Hugging Face hub.

    Parameters
    ----------
    model_name : str, default="facebook/esm2_t33_650M_UR50D"
        The identifier of the pre-trained ESM-2 model to load.

    Returns
    -------
    model : EsmForMaskedLM
        The loaded ESM-2 model moved to the appropriate device (CPU/GPU).
    tokenizer : EsmTokenizer
        The corresponding tokenizer for the model.
    device : torch.device
        The device ('cuda' or 'cpu') where the model is loaded.
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
    Generate ESM-2 embeddings and logits for a list of protein sequences in batches.

    Parameters
    ----------
    sequences : List[str]
        A list of protein sequences.
    model : EsmForMaskedLM
        The pre-trained ESM-2 model.
    tokenizer : EsmTokenizer
        The tokenizer corresponding to the ESM-2 model.
    device : torch.device
        The device to run inference on.
    batch_size : int, default=32
        Number of sequences to process in a single batch.
    max_length : int, default=1024
        Maximum sequence length for truncation/padding.
    extract_layers : List[int], optional, default=None
        Indices of layers to extract embeddings from. If None, extracts the last layer (-1).

    Returns
    -------
    sequence_data : Dict[str, Dict]
        A dictionary where keys are sequences and values are dictionaries containing:
        - 'logits': Tensor of logits for the sequence.
        - 'embeddings': Dictionary mapping layer indices to embedding tensors.
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


def go_esm_vs_unit_similarity(topic_model, df_go_to_uniprot, go_name2id, top_n_strategy='all', top_n=None, return_sims=False, profile_similarity=None):
    """
    Calculate the Spearman correlation between the GO term similarity matrix derived from
    PUMA unit representations (c-TF-IDF) and the "ground truth" similarity matrix derived from ESM-2 embeddings.

    Parameters
    ----------
    topic_model : BERTopic
        The trained BERTopic model containing the c-TF-IDF representations.
    df_go_to_uniprot : pd.DataFrame
        DataFrame indexed by GO ID, containing the aggregated 'esm_embedding' for each GO term.
    go_name2id : Dict[str, str]
        Mapping from GO term names (used in topic model) to GO IDs.
    top_n_strategy : str, default='all'
        Strategy for selecting unit features ('all', 'random', or None for top N).
    top_n : int, optional
        Number of top words to consider if strategy is not 'all'.
    return_sims : bool, default=False
        If True, returns the raw similarity matrices instead of the correlation score.
    profile_similarity : Any, optional
        Unused parameter kept for compatibility.

    Returns
    -------
    result : Dict or Tuple
        If return_sims is True:
            Tuple of (unit_similarity_matrix, esm_similarity_matrix).
        Else:
            Dictionary containing 'correlation_spearman_esm_unit' and 'p_value_spearman_esm_unit'.
    """
    # 1. Get topic information and corresponding GO terms from the trained model
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1]  # Exclude the outlier topic
    go_names = topic_info.CustomName.tolist()

    go_ids = [go_name2id[name] for name in go_names]
    topic_ids_in_model = topic_info.Topic.tolist()

    if len(topic_ids_in_model) != len(topic_info):
        print('Error in shape')
        return

    # 2. Compute the similarity matrix for protein unit representations (c-TF-IDF vectors)
    if top_n_strategy == 'all':
        topic_vectors = topic_model.c_tf_idf_[:, :]
    else:
        unit_vocab = topic_model.get_params()['vectorizer_model'].vocabulary_
        top_units_set = set(topic_info['Representation'].apply(lambda x: x[:top_n]).sum())
        if '' in top_units_set:
            top_units_set.remove('')

        if top_n_strategy == 'random':
            top_units_set = np.random.choice(list(unit_vocab.keys()), len(top_units_set), replace=False)

        top_units_indices = [unit_vocab[top_unit] for top_unit in top_units_set]
            
        topic_vectors = topic_model.c_tf_idf_[:, top_units_indices]

    
    unit_similarity_matrix = cosine_similarity(topic_vectors)
    esm_similarity_matrix = cosine_similarity(list(df_go_to_uniprot.loc[go_ids]['esm_embedding']))
    
    if return_sims:
        return unit_similarity_matrix, esm_similarity_matrix

    # 4. Correlate the two distance/similarity measures
    # Flatten the upper triangles of the matrices to get pairwise values (excluding the diagonal)
    indices = np.triu_indices_from(unit_similarity_matrix, k=1)
    unit_sim_flat = unit_similarity_matrix[indices]
    esm_sim_flat = esm_similarity_matrix[indices]

    # Filter out any pairs where the DAG distance could not be calculated
    valid_indices = ~np.isnan(unit_sim_flat)
    unit_sim_flat = unit_sim_flat[valid_indices]
    esm_sim_flat = esm_sim_flat[valid_indices]

    # Calculate Spearman correlation
    correlation_spearman_esm_unit, p_value_spearman_esm_unit = spearmanr(esm_sim_flat, unit_sim_flat)

    return {
        'correlation_spearman_esm_unit': correlation_spearman_esm_unit, 'p_value_spearman_esm_unit': p_value_spearman_esm_unit,
        }


def split_go_evidence(df, threshold=None, split_ratio=0, denom=1, random_state=42):
    """
    Balance the distribution of GO terms by undersampling overrepresented terms.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing protein-GO associations.
    threshold : int, optional
        The maximum number of proteins allowed per GO term. If None, calculated based on mean frequency.
    split_ratio : float, default=0
        Ratio for splitting into train/test sets. If 0, no splitting is performed.
    denom : int, default=1
        Denominator to adjust the automatic threshold calculation.
    random_state : int, default=42
        Seed for random sampling reproducibility.

    Returns
    -------
    df_balanced : pd.DataFrame
        The balanced DataFrame (or training set if split_ratio > 0).
    df_test : pd.DataFrame or None
        The test set if split_ratio > 0, else None.
    """
    if threshold is None:
        threshold = df['go_name'].value_counts()[:df['go_name'].nunique()//denom].mean().astype(int)
    go_counts = df['go_name'].value_counts()

    overrepresented_go_names = go_counts[go_counts > threshold].index.tolist()
    underrepresented_go_names = go_counts[go_counts <= threshold].index.tolist()

    supplement_over_rows = []
    for go_name in overrepresented_go_names:
        # Bu go_name'e sahip proteinlerden sadece ilk satırlarını seç (random protein seçimi)
        candidate_proteins = df[df['go_name'] == go_name]
        
        # Eğer yeterli sayıda protein yoksa, olan kadarını al
        if random_state is not None:
            sampled = candidate_proteins.sample(n=threshold, random_state=random_state)
        else:
            sampled = candidate_proteins.sample(n=threshold)
        
        supplement_over_rows.append(sampled)

    df_balanced = pd.concat([pd.concat(supplement_over_rows), df[df['go_name'].isin(underrepresented_go_names)]])

    if split_ratio == 0:
        return df_balanced, None
    else:
        df_train, df_test = split_data(df_balanced, split_ratio)
        return df_train, df_test
    

def scale_matrix(A):
    """
    Apply Min-Max scaling to the strictly lower triangle of a matrix for heatmap visualization.
    The resulting matrix contains NaNs on the diagonal and upper triangle.

    Parameters
    ----------
    A : np.ndarray
        The input square matrix.

    Returns
    -------
    A_scaled : np.ndarray
        The scaled matrix with NaNs in the upper triangle and diagonal.
    """
    # --- Process the matrix for heatmap ---

    # 2. Get the indices for the strictly lower triangle (k=-1 excludes the diagonal)
    n = A.shape[0]
    r, c = np.tril_indices(n, k=-1)

    # 3. Apply Min-Max scaling to the lower triangle values

    # Extract the values from the strictly lower triangle
    lower_tri_values = A[r, c]

    # Calculate Min-Max
    min_val = lower_tri_values.min()
    max_val = lower_tri_values.max()

    # Min-Max Scaling Formula: X_scaled = (X - min) / (max - min)
    if max_val == min_val:
        # Handle edge case: all values are the same
        scaled_values = np.zeros_like(lower_tri_values, dtype=float)
    else:
        scaled_values = (lower_tri_values - min_val) / (max_val - min_val)

    # 4. Create the final matrix for plotting and the mask

    # Initialize the matrix for the heatmap with NaNs.
    # NaNs are best for masking as seaborn ignores them.
    A_scaled = np.full((n, n), np.nan)

    # Place the scaled values back into the strictly lower triangle
    A_scaled[r, c] = scaled_values

    return A_scaled


def create_unit_documents(df: pd.DataFrame, tokenizer_col: str, token_len_thr: int = 0) -> List[str]:
    """
    Converts tokenizer outputs for each protein into a space-separated document format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the protein data.
    tokenizer_col : str
        The name of the column containing the tokenized sequences (list of units).
    token_len_thr : int, default=0
        The minimum length of a token to be included in the document.

    Returns
    -------
    documents : List[str]
        A list of strings, where each string is a space-separated sequence of protein units.
    """
    return df[tokenizer_col].apply(lambda units: ' '.join(unit for unit in units if len(unit) >= token_len_thr)).tolist()


def create_go_labels(df: pd.DataFrame, go_col: str = 'go_id') -> List[List[str]]:
    """
    Extracts GO terms for each protein into a list format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the protein data.
    go_col : str, default='go_id'
        The name of the column containing the GO term labels.

    Returns
    -------
    labels : List[List[str]]
        A list of lists, where each inner list contains the GO terms (or single term) for a protein.
    """
    return df[go_col].tolist()

# Function to create a standard BERTopic model for manual topic modeling
def create_bertopic_model(documents: List[str], go_labels: List[str], token_len_thr: int = 0, top_n_words: int = 10, max_df = 0.9, min_df = 1) -> Tuple[BERTopic, np.ndarray]:
    """
    Creates and fits a standard BERTopic model for manual topic modeling using supervised GO labels.
    Uses ClassTfidfTransformer to generate functional representations (c-TF-IDF vectors).

    Parameters
    ----------
    documents : List[str]
        A list of protein sequences represented as documents.
    go_labels : List[str]
        A list of GO labels corresponding to each document (for supervised modeling).
    token_len_thr : int, default=0
        The minimum length of a token to be included in the vocabulary.
    top_n_words : int, default=10
        The number of top words per topic to extract.
    max_df : float or int, default=0.9
        When building the vocabulary, ignore terms with a document frequency strictly higher than this.
    min_df : float or int, default=1
        When building the vocabulary, ignore terms with a document frequency strictly lower than this.

    Returns
    -------
    topic_model : BERTopic
        The trained BERTopic model.
    topics : np.ndarray
        The array of assigned topics (derived from GO labels).
    """
    
    # One-hot encoding for GO terms
    lb = LabelBinarizer()
    go_binary = lb.fit_transform(go_labels)
    
    # CountVectorizer settings for protein units
    vectorizer_model = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w{%d,}\b" % token_len_thr,
        stop_words=None,
        ngram_range=(1, 1),
        max_df=max_df,
        min_df=min_df,
    )
    
    # Custom TF-IDF for protein unit importance
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    
    # Models for manual topic assignment
    empty_embedding_model = BaseEmbedder()
    empty_dimensionality_model = BaseDimensionalityReduction()
    empty_cluster_model = BaseCluster()

    # Create BERTopic model without a specified embedding model for manual topic modeling
    topic_model = BERTopic(
        top_n_words=top_n_words,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        embedding_model=empty_embedding_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=empty_cluster_model,
        verbose=False
    )
    
    # Manually assign topics based on GO terms
    topics = np.argmax(go_binary, axis=1)
    topics[np.sum(go_binary, axis=1) == 0] = -1  # Mark proteins with no GO term as outliers

    # Fit the model with the documents and manually assigned topics
    topic_model.fit_transform(documents, y=topics)
    
    # Match topic names with GO term IDs
    mappings = topic_model.topic_mapper_.get_mappings()
    mappings = {value: lb.classes_[key] for key, value in mappings.items()}
    topic_model.set_topic_labels(mappings)
    
    return topic_model, topics



def build_unit_similarity_matrix(unit_relationships: Dict[str, Dict],
                                 vocabulary: List[str],
                                 alpha: float = 1.0,
                                 beta: float = 3.0,
                                 theta: float = 2.0) -> sp.spmatrix:
    """
    Builds a SPARSE adjacency matrix A representing genealogical relationships between protein units.
    
    Relationships encoded:
    - Hierarchical (Parent-Child): Weight = alpha
    - Mutational (Sibling): Weight = beta
    - Mutational (Parent-Child): Weight = theta

    Parameters
    ----------
    unit_relationships : Dict[str, Dict]
        A dictionary containing 'hierarchical' and 'mutational' dictionaries mapping parents to children.
    vocabulary : List[str]
        A list of all protein units in the vocabulary (the nodes of the graph).
    alpha : float, default=1.0
        Weight for hierarchical parent-child relationships.
    beta : float, default=3.0
        Weight for sibling relationships (child-child within same family).
    theta : float, default=2.0
        Weight for mutational parent-child relationships.

    Returns
    -------
    A : sp.spmatrix
        A square sparse adjacency matrix (CSR format) representing unit similarities.
    """
    
    # unit_to_idx = {unit: idx for idx, unit in enumerate(vocabulary)}
    unit_to_idx = vocabulary

    n_units = len(vocabulary)
    
    # Use a LIL (List of Lists) matrix for efficient incremental construction
    A = sp.lil_matrix((n_units, n_units))
    
    # Add hierarchical relationships (parent-child)
    if 'hierarchical' in unit_relationships:
        for parent, children in unit_relationships['hierarchical'].items():
            if parent in unit_to_idx:
                parent_idx = unit_to_idx[parent]
                for child in children:
                    if child in unit_to_idx:
                        child_idx = unit_to_idx[child]
                        A[parent_idx, child_idx] = alpha
                        A[child_idx, parent_idx] = alpha
    
    # Add mutational relationships (family-based)
    if 'mutational' in unit_relationships:
        for parent, children in unit_relationships['mutational'].items():
            if parent in unit_to_idx:
                parent_idx = unit_to_idx[parent]
                for i in range(len(children)):
                    if children[i] in unit_to_idx:
                        child_i_idx = unit_to_idx[children[i]]
                        A[parent_idx, child_i_idx] = theta
                        A[child_i_idx, parent_idx] = theta
                        for j in range(i + 1, len(children)):
                            if children[j] in unit_to_idx:
                                child_j_idx = unit_to_idx[children[j]]
                                A[child_i_idx, child_j_idx] = beta
                                A[child_j_idx, child_i_idx] = beta
    
    # Convert to CSR (Compressed Sparse Row) format for efficient
    # matrix multiplication in the next steps
    return A.tocsr()


def compute_smoothing_matrix(similarity_matrix: sp.spmatrix, 
                           lambda_smooth: float = 0.5) -> Optional[sp.spmatrix]:
    """
    Precomputes the SPARSE smoothing matrix S = (I + λA) for graph-based smoothing.
    Normalizes the adjacency matrix A before combining.

    Parameters
    ----------
    similarity_matrix : sp.spmatrix
        A square sparse matrix A representing similarity between units.
    lambda_smooth : float, default=0.5
        The smoothing parameter λ governing the influence of the graph.

    Returns
    -------
    smoothing_matrix : sp.spmatrix
        The computed sparse smoothing matrix (CSR format).
    """
    if similarity_matrix is None or similarity_matrix.sum() == 0:
        return None
    
    # Copy and convert to CSC (Compressed Sparse Column) for efficient
    # column-wise operations (summing, normalization)
    A = similarity_matrix.copy().tocsc()
    
    # Column Normalization (sparse-safe)
    col_sums = np.array(A.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1
    
    # Create a sparse diagonal matrix of the inverse sums
    inv_col_sums = 1.0 / col_sums
    D_inv = sp.diags(inv_col_sums)
    
    # Normalize A by post-multiplying with D_inv (A_norm = A * D_inv)
    A_norm = A.dot(D_inv)
    
    # Create a sparse identity matrix
    I = sp.eye(A_norm.shape[0], format='csc')
    
    # This is now all sparse matrix arithmetic
    smoothing_matrix = (1 - lambda_smooth) * I + lambda_smooth * A_norm
    
    # Convert back to CSR for efficient row-wise dot products (X.dot(S))
    return smoothing_matrix.tocsr()


def apply_graph_smoothing(X: sp.csr_matrix, 
                        smoothing_matrix: sp.spmatrix) -> Union[np.ndarray, sp.csr_matrix]:
    """
    Applies graph smoothing to a document-term matrix using the formula D' = D * S.

    Parameters
    ----------
    X : sp.csr_matrix or np.ndarray
        The document-term matrix (n_documents x n_features).
    smoothing_matrix : sp.spmatrix
        The precomputed sparse smoothing matrix S.

    Returns
    -------
    X_smoothed : sp.csr_matrix or np.ndarray
        The smoothed document-term matrix.
    """
    if smoothing_matrix is None:
        return X
    
    # This is now a fast sparse-sparse dot product (if X is sparse)
    # or a sparse-dense dot product (if X is dense)
    if sp.issparse(X):
        # Removed the unnecessary and slow `sp.csr_matrix(smoothing_matrix)` conversion
        X_smoothed = X.dot(smoothing_matrix)
    else:
        # This handles the case if X is dense.
        # We must convert the sparse smoothing_matrix to a dense array for this.
        X_smoothed = np.dot(X, smoothing_matrix.toarray())
        
    return X_smoothed



class GraphAwareCountVectorizer(CountVectorizer):
    """
    A custom CountVectorizer that applies graph smoothing to the document-term matrix
    immediately after vectorization.

    Parameters
    ----------
    similarity_matrix : sp.spmatrix, optional
        A matrix representing the similarity between vocabulary items (units).
    lambda_smooth : float, default=0.5
        The smoothing parameter λ.
    **kwargs :
        Standard CountVectorizer arguments.
    """
    def __init__(self,
                 # Type hint updated to accept sparse matrices
                 similarity_matrix: Optional[sp.spmatrix] = None, 
                 lambda_smooth: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.lambda_smooth = lambda_smooth
        self.similarity_matrix = similarity_matrix
        self.smoothing_matrix = None # This will now store a sparse matrix

    def fit_transform(self, raw_documents: List[str], y: Any = None) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Fits the vectorizer, computes the smoothing matrix, and applies it to the transformed matrix.

        Parameters
        ----------
        raw_documents : List[str]
            A list of documents to be transformed.
        y : Any, optional
            Ignored. Present for API consistency.

        Returns
        -------
        X : sp.csr_matrix
            The smoothed document-term matrix.
        """
        X = super().fit_transform(raw_documents, y)
        
        # This now calls the sparse-aware function
        self.smoothing_matrix = compute_smoothing_matrix(self.similarity_matrix, self.lambda_smooth)
        
        if self.smoothing_matrix is not None:
            # This now calls the optimized sparse-dot function
            X = apply_graph_smoothing(X, self.smoothing_matrix)

        return X

    def transform(self, raw_documents: List[str]) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Transforms documents and applies the precomputed graph smoothing.

        Parameters
        ----------
        raw_documents : List[str]
            A list of documents to be transformed.

        Returns
        -------
        X : sp.csr_matrix
            The smoothed document-term matrix.
        """
        X = super().transform(raw_documents)
        
        if self.smoothing_matrix is not None:
            X = apply_graph_smoothing(X, self.smoothing_matrix)
            
        return X

def create_graph_aware_bertopic_model(documents: List[str], 
                                      go_labels: List[str],
                                      unit_relationships: Optional[Dict[str, Dict]] = None,
                                      token_len_thr: int = 0,
                                      top_n_words: int = 10,
                                      max_df = 0.9,
                                      min_df = 1,
                                      lambda_smooth: float = 0.5,
                                      alpha: float = 1.0,
                                      beta: float = 3.0,
                                      theta: float = 2.0,
                                      to_shuffle: bool = False) -> Tuple[BERTopic, np.ndarray, np.ndarray]:
    """
    Creates and fits a Graph-Aware BERTopic model.
    This model constructs a unit vocabulary, builds a similarity graph based on PUMA genealogy,
    and applies graph smoothing to the term frequency matrix before c-TF-IDF calculation.

    Parameters
    ----------
    documents : List[str]
        A list of protein sequences represented as documents.
    go_labels : List[str]
        A list of GO labels for each document.
    unit_relationships : Dict[str, Dict], optional
        A dictionary of unit relationships (hierarchical, mutational) for building the similarity matrix.
    token_len_thr : int, default=0
        The minimum length of a token to be included.
    top_n_words : int, default=10
        The number of words per topic to extract.
    max_df : float or int, default=0.9
        Corpus-specific stop word threshold.
    min_df : float or int, default=1
        Minimum document frequency cut-off.
    lambda_smooth : float, default=0.5
        The graph smoothing parameter λ.
    alpha : float, default=1.0
        Weight for hierarchical relationships.
    beta : float, default=3.0
        Weight for sibling relationships.
    theta : float, default=2.0
        Weight for mutational relationships.
    to_shuffle : bool, default=False
        If True, shuffles the similarity matrix (used for null hypothesis testing).

    Returns
    -------
    topic_model : BERTopic
        The trained Graph-Aware BERTopic model.
    topics : np.ndarray
        The array of assigned topics.
    similarity_matrix : sp.spmatrix
        The computed unit similarity matrix used for smoothing.
    """
    
    lb = LabelBinarizer()
    go_binary = lb.fit_transform(go_labels)
    
    temp_topic_model, _ = create_bertopic_model(documents, go_labels, token_len_thr, top_n_words, max_df, min_df)
    vocabulary = temp_topic_model.get_params()['vectorizer_model'].vocabulary_

    similarity_matrix = None
    if unit_relationships is not None:
        similarity_matrix = build_unit_similarity_matrix(
            unit_relationships=unit_relationships,
            vocabulary=vocabulary,
            alpha=alpha,
            beta=beta,
            theta=theta
        )
        # print(f"Built similarity matrix with shape: {similarity_matrix.shape}")
        # print(f"Non-zero edges: {np.sum(similarity_matrix > 0)}")
    if to_shuffle:
        np.random.shuffle(similarity_matrix)
    
    vectorizer_model = GraphAwareCountVectorizer(
        similarity_matrix=similarity_matrix,
        lambda_smooth=lambda_smooth,
        lowercase=False,
        token_pattern=r"(?u)\b\w{%d,}\b" % token_len_thr,
        stop_words=None,
        ngram_range=(1, 1),
        vocabulary=vocabulary
    )
    
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    empty_embedding_model = BaseEmbedder()
    empty_dimensionality_model = BaseDimensionalityReduction()
    empty_cluster_model = BaseCluster()
    
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        embedding_model=empty_embedding_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=empty_cluster_model,
        top_n_words=top_n_words,
        verbose=False
    )
    
    topics = np.argmax(go_binary, axis=1)
    topics[np.sum(go_binary, axis=1) == 0] = -1
    
    topic_model.fit_transform(documents, y=topics)

    # Match topic names with GO term IDs
    mappings = topic_model.topic_mapper_.get_mappings()
    mappings = {value: lb.classes_[key] for key, value in mappings.items()}
    topic_model.set_topic_labels(mappings)
    
    # print(f"Model trained with {len(set(topics))} topics")
    # print(f"Graph smoothing parameter λ = {lambda_smooth}")
    
    return topic_model, topics, similarity_matrix