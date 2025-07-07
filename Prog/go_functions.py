import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import itertools
import json
import random
from collections import Counter
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Any, Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired

import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from goatools.gosubdag.gosubdag import GoSubDag

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.color_palette("Spectral", as_cmap=True)

colors = ['olivedrab', 'lightseagreen', 'slateblue', 'gold', 'cornflowerblue', 'darkred']
markers = ["o--", "d--", "*--", "X--", "P--", "p--"]
linestyles = ['-', '--', '-.', ':', "."]

label_pad = 6
title_pad = 15
title_size = 16
tick_size = 14

plt.rcParams["figure.figsize"] = (8, 8)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=19)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=13)
plt.rcParams['figure.dpi'] = 100


# Helper function to convert protein units into a document format
def create_unit_documents(df: pd.DataFrame, tokenizer_col: str, token_len_thr: int = 0) -> List[str]:
    """
    Converts tokenizer outputs for each protein into a space-separated document format.

    Args:
        df: DataFrame containing the protein data.
        tokenizer_col: The name of the column containing the tokenized sequences.
        token_len_thr: The minimum length of a token to be included.

    Returns:
        A list of strings, where each string represents a document.
    """
    return df[tokenizer_col].apply(lambda units: ' '.join(unit for unit in units if len(unit) >= token_len_thr)).tolist()

# Helper function to create GO labels from the dataframe
def create_go_labels(df: pd.DataFrame, go_col: str = 'go_id') -> List[List[str]]:
    """
    Extracts GO terms for each protein into a list format.

    Args:
        df: DataFrame containing the protein data.
        go_col: The name of the column containing the GO term labels.

    Returns:
        A list of lists, where each inner list contains the GO terms for a protein.
    """
    return df[go_col].tolist()

# Function to create a standard BERTopic model for manual topic modeling
def create_bertopic_model(documents: List[str], go_labels: List[str], token_len_thr: int = 0, top_n_words: int = 10) -> Tuple[BERTopic, np.ndarray]:
    """
    Creates and fits a standard BERTopic model for manual topic modeling based on GO labels.

    Args:
        documents: A list of protein sequences represented as documents.
        go_labels: A list of GO labels corresponding to each document.
        token_len_thr: The minimum length of a token to be included.
        top_n_words: The number of words per topic to extract. 

    Returns:
        A tuple containing the trained BERTopic model and the assigned topics array.
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
        min_df=3,
        max_df=0.8,
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

def compute_smoothing_matrix(similarity_matrix: np.ndarray, lambda_smooth: float = 0.1) -> Optional[np.ndarray]:
    """
    Precomputes the smoothing matrix (I + λA) for graph-based smoothing.

    Args:
        similarity_matrix: A square matrix representing the similarity between units.
        lambda_smooth: The smoothing parameter (lambda).

    Returns:
        The computed smoothing matrix, or None if the input matrix is empty.
    """
    if similarity_matrix is None or similarity_matrix.sum() == 0:
        return None
    
    A = similarity_matrix.copy()
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    A_norm = A / row_sums[:, np.newaxis]
    
    I = np.eye(A_norm.shape[0])
    smoothing_matrix = I + lambda_smooth * A_norm
    
    return smoothing_matrix

def apply_graph_smoothing(X: sp.csr_matrix, smoothing_matrix: np.ndarray) -> Union[np.ndarray, sp.csr_matrix]:
    """
    Applies graph smoothing to a document-term matrix.

    Args:
        X: The document-term matrix (n_documents x n_features).
        smoothing_matrix: The precomputed smoothing matrix.

    Returns:
        The smoothed document-term matrix.
    """
    if smoothing_matrix is None:
        return X
    
    if sp.issparse(X):
        X_smoothed = X.dot(sp.csr_matrix(smoothing_matrix))
    else:
        X_smoothed = np.dot(X, smoothing_matrix)
        
    return X_smoothed

class GraphAwareCountVectorizer(CountVectorizer):
    """
    A custom CountVectorizer that applies graph smoothing to the document-term matrix.

    Args:
        similarity_matrix: A matrix representing the similarity between vocabulary items.
        lambda_smooth: The smoothing parameter for graph smoothing.
    """
    def __init__(self, 
                 similarity_matrix: Optional[np.ndarray] = None,
                 lambda_smooth: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.lambda_smooth = lambda_smooth
        self.similarity_matrix = similarity_matrix
        self.smoothing_matrix = None
    
    def fit_transform(self, raw_documents: List[str], y: Any = None) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Fits the vectorizer and applies graph smoothing to the transformed matrix.

        Args:
            raw_documents: A list of documents to be transformed.
            y: Ignored. Present for API consistency.

        Returns:
            The smoothed document-term matrix.
        """
        X = super().fit_transform(raw_documents, y)
        self.smoothing_matrix = compute_smoothing_matrix(self.similarity_matrix, self.lambda_smooth)
        
        if self.smoothing_matrix is not None:
            X = apply_graph_smoothing(X, self.smoothing_matrix)
            
        return X
    
    def transform(self, raw_documents: List[str]) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Transforms documents and applies graph smoothing.

        Args:
            raw_documents: A list of documents to be transformed.

        Returns:
            The smoothed document-term matrix.
        """
        X = super().transform(raw_documents)
        
        if self.smoothing_matrix is not None:
            X = apply_graph_smoothing(X, self.smoothing_matrix)
            
        return X

def build_unit_similarity_matrix(unit_relationships: Dict[str, Dict], 
                                 vocabulary: List[str],
                                 alpha: float = 1.0, 
                                 beta: float = 0.5,
                                 theta: float = 0.7) -> np.ndarray:
    """
    Builds an adjacency matrix representing relationships between protein units.

    Args:
        unit_relationships: A dictionary containing hierarchical and mutational relationships.
        vocabulary: A list of all protein units in the vocabulary.
        alpha: Weight for hierarchical parent-child relationships.
        beta: Weight for sibling relationships.
        theta: Weight for mutational parent-child relationships.

    Returns:
        A square adjacency matrix representing unit similarities.
    """
    
    # unit_to_idx = {unit: idx for idx, unit in enumerate(vocabulary)}
    unit_to_idx = vocabulary
    n_units = len(vocabulary)
    A = np.zeros((n_units, n_units))
    
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
    
    return A

def create_graph_aware_bertopic_model(documents: List[str], 
                                      go_labels: List[str],
                                      unit_relationships: Optional[Dict[str, Dict]] = None,
                                      token_len_thr: int = 0,
                                      top_n_words: int = 10,
                                      lambda_smooth: float = 0.1,
                                      alpha: float = 1.0,
                                      beta: float = 0.5,
                                      theta: float = 0.7) -> Tuple[BERTopic, np.ndarray, np.ndarray]:
    """
    Creates and fits a Graph-Aware BERTopic model.

    Args:
        documents: A list of protein sequences represented as documents.
        go_labels: A list of GO labels for each document.
        unit_relationships: A dictionary of unit relationships for building the similarity matrix.
        token_len_thr: The minimum length of a token to be included.
        top_n_words: The number of words per topic to extract. 
        lambda_smooth: The graph smoothing parameter.
        alpha: Weight for hierarchical relationships.
        beta: Weight for sibling relationships.
        theta: Weight for mutational relationships.

    Returns:
        A tuple containing the trained model, topics array, and the similarity matrix.
    """
    
    lb = LabelBinarizer()
    go_binary = lb.fit_transform(go_labels)
    
    temp_topic_model, _ = create_bertopic_model(documents, go_labels, token_len_thr, top_n_words)
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

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing sets based on unique UniProt IDs.

    Args:
        df: The DataFrame to split.
        test_size: The proportion of the dataset to allocate to the test set.
        random_state: The seed for the random number generator.

    Returns:
        A tuple containing the training and testing DataFrames.
    """
    unique_uniprot_ids = df["uniprot_id"].unique().tolist()
    random.shuffle(unique_uniprot_ids)

    train_ids, test_ids = train_test_split(unique_uniprot_ids, test_size=test_size, random_state=random_state)

    df_train = df[df["uniprot_id"].isin(train_ids)].copy()
    df_test = df[df["uniprot_id"].isin(test_ids)].copy()

    train_dist = df_train['go_id'].value_counts(normalize=True)
    test_dist = df_test['go_id'].value_counts(normalize=True)

    distribution_df = pd.DataFrame({
        "train_ratio": train_dist,
        "test_ratio": test_dist
    }).fillna(0)
    distribution_df["difference"] = abs(distribution_df["train_ratio"] - distribution_df["test_ratio"])

    # print("GO term distribution difference between train and test sets:")
    # print(distribution_df.sort_values("difference", ascending=False).head(5))
    # print()

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def substrings_with_overlap(s: str, min_len: int = 4, max_len: int = 10) -> List[str]:
    """
    Generates all substrings with overlap within a given length range.

    Args:
        s: The input string.
        min_len: The minimum length of substrings.
        max_len: The maximum length of substrings.

    Returns:
        A list of all overlapping substrings.
    """
    results = []
    for i in range(len(s)):
        for l in range(min_len, max_len + 1):
            if i + l <= len(s):
                results.append(s[i:i+l])
    return results

def evaluate_go_term_representations(df_test: pd.DataFrame, topic_model: BERTopic, tokenizer_col: str, go_col: str = 'go_name',
                                     raw_sequence_col: str = 'sequence', raw_or_tokenized: str = 'tokenized', scoring_type: str = 'exist') -> Dict[str, Any]:
    """
    Evaluates learned GO term representations on the test set using single-label metrics.

    Args:
        df_test: The test DataFrame.
        topic_model: The trained BERTopic model.
        tokenizer_col: The column with tokenized sequences.
        go_col: The column with GO term labels.
        raw_sequence_col: The column with raw protein sequences.
        raw_or_tokenized: Specifies whether to use 'raw' sequences or 'tokenized' units.
        scoring_type: The scoring method ('exist' or 'freq').

    Returns:
        A dictionary containing single-label evaluation results.
    """

    print(f"Evaluation mode: {raw_or_tokenized}, Scoring type: {scoring_type}")
    
    go_term_representations = {}
    topic_info = topic_model.get_topic_info()
    topic_id_to_go_name = {row.Topic: row.CustomName for _, row in topic_info.iterrows()}

    for topic_id, go_name in topic_id_to_go_name.items():
        if topic_id == -1:
            continue
        representation = topic_model.get_topic(topic_id)
        if representation:
            go_term_representations[go_name] = {unit: score for unit, score in representation}

    y_true = []
    all_sorted_predictions = []
    reciprocal_ranks = []
    
    test_proteins = df_test.groupby('uniprot_id')

    for uniprot_id, protein_data in tqdm(test_proteins, desc="Evaluating Test Set (Single-Label)"):
        true_go_name = protein_data.iloc[0][go_col]
        
        if raw_or_tokenized == 'tokenized':
            tokenized_sequence = protein_data.iloc[0][tokenizer_col]
            token_counts = Counter(tokenized_sequence) if tokenized_sequence else Counter()
        else:
            raw_sequence = protein_data.iloc[0][raw_sequence_col]
            token_counts = Counter(substrings_with_overlap(raw_sequence)) if raw_sequence else Counter()
            
        if not token_counts:
            continue

        if scoring_type == 'freq':
            go_scores = {go_name: sum(score * token_counts.get(unit, 0) for unit, score in rep.items()) for go_name, rep in go_term_representations.items()}
        else:
            go_scores = {go_name: sum(score for unit, score in rep.items() if unit in token_counts) for go_name, rep in go_term_representations.items()}
            # go_scores = {go_name: sum(len(rep)-idx for idx, (unit, score) in enumerate(rep.items()) if unit in token_counts) for go_name, rep in go_term_representations.items()}

        if not go_scores:
            continue

        sorted_go_terms = sorted(go_scores.items(), key=lambda item: item[1], reverse=True)
        sorted_prediction_names = [go_name for go_name, score in sorted_go_terms]
        all_sorted_predictions.append(sorted_prediction_names)
        y_true.append(true_go_name)
        
        try:
            rank = sorted_prediction_names.index(true_go_name) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

    results = {}
    if not y_true:
        return {"error": "No test data to evaluate."}

    accuracy_at_k, precision_at_k, recall_at_k = {}, {}, {}
    for k in [1, 3, 5, 10]:
        p_scores, r_scores = [], []
        for i, true_label in enumerate(y_true):
            pred_set_at_k = set(all_sorted_predictions[i][:k])
            hits = 1 if true_label in pred_set_at_k else 0
            p_scores.append(hits / k)
            r_scores.append(hits)
        
        accuracy_at_k[f'accuracy_at_{k}'] = np.mean(r_scores) if r_scores else 0
        precision_at_k[f'precision_at_{k}'] = np.mean(p_scores) if p_scores else 0
        recall_at_k[f'recall_at_{k}'] = np.mean(r_scores) if r_scores else 0
        
    results.update({'accuracy_at_k': accuracy_at_k, 'precision_at_k': precision_at_k, 'recall_at_k': recall_at_k})
    
    results['mean_reciprocal_rank'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    y_pred = [preds[0] if preds else None for preds in all_sorted_predictions]
    valid_indices = [i for i, p in enumerate(y_pred) if p is not None]
    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]

    if y_true_filtered and y_pred_filtered:
        labels = sorted(list(set(y_true_filtered) | set(y_pred_filtered)))
        report = classification_report(y_true_filtered, y_pred_filtered, labels=labels, output_dict=True, zero_division=0)
        
        results['classification_summary'] = {
            'accuracy': report['accuracy'],
            # 'micro_avg': report['micro avg'],
            'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg']
        }
        results['per_go_term_metrics'] = {label: report[label] for label in labels if label in report}

    return results

def evaluate_go_term_representations_multilabel(df_test: pd.DataFrame, topic_model: BERTopic, tokenizer_col: str, go_col: str = 'go_name',
                                                raw_sequence_col: str = 'sequence', raw_or_tokenized: str = 'tokenized', scoring_type: str = 'exist') -> Dict[str, Any]:
    """
    Evaluates GO term representations using multi-label metrics.

    Args:
        df_test: The test DataFrame.
        topic_model: The trained BERTopic model.
        tokenizer_col: The column with tokenized sequences.
        go_col: The column with GO term labels.
        raw_sequence_col: The column with raw protein sequences.
        raw_or_tokenized: Specifies whether to use 'raw' sequences or 'tokenized' units.
        scoring_type: The scoring method ('exist' or 'freq').

    Returns:
        A dictionary containing multi-label evaluation results.
    """

    print(f"Evaluation mode: {raw_or_tokenized}, Scoring type: {scoring_type}")
    
    go_term_representations = {}
    topic_info = topic_model.get_topic_info()
    topic_id_to_go_name = {row.Topic: row.CustomName for _, row in topic_info.iterrows()}

    for topic_id, go_name in topic_id_to_go_name.items():
        if topic_id == -1:
            continue
        representation = topic_model.get_topic(topic_id)
        if representation:
            go_term_representations[go_name] = {unit: score for unit, score in representation}

    y_true_sets = []
    all_sorted_predictions = []
    
    test_proteins = df_test.groupby('uniprot_id')

    for uniprot_id, protein_data in tqdm(test_proteins, desc="Evaluating Test Set (Multi-Label)"):
        true_go_names = set(protein_data[go_col].tolist())

        if raw_or_tokenized == 'tokenized':
            tokenized_sequence = protein_data.iloc[0][tokenizer_col]
            token_counts = Counter(tokenized_sequence) if tokenized_sequence else Counter()
        else:
            raw_sequence = protein_data.iloc[0][raw_sequence_col]
            token_counts = Counter(substrings_with_overlap(raw_sequence)) if raw_sequence else Counter()
            
        if not token_counts:
            continue

        if scoring_type == 'freq':
            go_scores = {go_name: sum(score * token_counts.get(unit, 0) for unit, score in rep.items()) for go_name, rep in go_term_representations.items()}
        else:
            go_scores = {go_name: sum(score for unit, score in rep.items() if unit in token_counts) for go_name, rep in go_term_representations.items()}
        
        if not go_scores:
            continue

        sorted_prediction_names = [go_name for go_name, score in sorted(go_scores.items(), key=lambda item: item[1], reverse=True)]
        all_sorted_predictions.append(sorted_prediction_names)
        y_true_sets.append(true_go_names)

    results = {}
    if not y_true_sets:
        return {"error": "No test data to evaluate."}

    reciprocal_ranks = []
    for i, true_set in enumerate(y_true_sets):
        rank = next((r + 1 for r, p in enumerate(all_sorted_predictions[i]) if p in true_set), 0)
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
    results['mean_reciprocal_rank'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    accuracy_at_k, precision_at_k, recall_at_k = {}, {}, {}
    for k in [1, 3, 5, 10]:
        acc_scores, p_scores, r_scores = [], [], []
        for i, true_set in enumerate(y_true_sets):
            if not true_set: continue
            pred_set_at_k = set(all_sorted_predictions[i][:k])
            hits = len(true_set.intersection(pred_set_at_k))
            acc_scores.append(1.0 if hits > 0 else 0.0)
            p_scores.append(hits / k)
            r_scores.append(hits / len(true_set))
        accuracy_at_k[f'accuracy_at_{k}'] = np.mean(acc_scores) if acc_scores else 0
        precision_at_k[f'precision_at_{k}'] = np.mean(p_scores) if p_scores else 0
        recall_at_k[f'recall_at_{k}'] = np.mean(r_scores) if r_scores else 0
    results.update({'accuracy_at_k': accuracy_at_k, 'precision_at_k': precision_at_k, 'recall_at_k': recall_at_k})

    all_labels = sorted(list(go_term_representations.keys()))
    mlb = MultiLabelBinarizer(classes=all_labels)
    y_true_binary = mlb.fit_transform(y_true_sets)
    y_pred_binary = [mlb.transform([all_sorted_predictions[i][:len(ts)]])[0] for i, ts in enumerate(y_true_sets) if len(ts) > 0]

    if y_pred_binary:
        y_true_binary_filtered = [y_true_binary[i] for i, ts in enumerate(y_true_sets) if len(ts) > 0]
        report = classification_report(y_true_binary_filtered, np.array(y_pred_binary), target_names=all_labels, output_dict=True, zero_division=0)
        results['classification_summary'] = {
            'micro_avg': report['micro avg'], 'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg'], 'samples_avg': report['samples avg']
        }
        results['per_go_term_metrics'] = {label: report[label] for label in all_labels if label in report}

    return results

def visualize_comparison_results(df_results: pd.DataFrame, methods: List[str], methods2names: Dict[str, str], vocab_sizes: List[int], figures_folder: str = 'figures'):
    """
    Generates and saves line plots to visualize model comparison results.

    Args:
        df_results: DataFrame containing the performance metrics.
        methods: A list of base method names to be plotted.
        methods2names: A dictionary mapping method names to display names.
        vocab_sizes: A list of vocabulary sizes used in the experiment.
        figures_folder: The directory where plots will be saved.
    """
    sns.set_style("whitegrid")

    for metric in df_results.columns[2:]:
        plt.figure()
        for i, method in enumerate(methods):
            method_name = methods2names.get(method, method)
            # Standard BERTopic
            plt.plot(vocab_sizes, [df_results.loc[(df_results['Tokenizer'] == f'{method} {vs}') & (df_results['Model'] == 'Standard BERTopic'), metric].iloc[0] for vs in vocab_sizes],
                     markers[i], color=colors[i], markersize=10, linestyle='-', label=f'{method_name} (Standard)')
            # Graph-Aware BERTopic
            plt.plot(vocab_sizes, [df_results.loc[(df_results['Tokenizer'] == f'{method} {vs}') & (df_results['Model'] == 'Graph-Aware BERTopic'), metric].iloc[0] for vs in vocab_sizes],
                    markers[i], color=colors[i], markersize=10, linestyle='--', label=f'{method_name} (Graph-Aware)')

        plt.title(f'Model Performance vs. Vocabulary Size: {metric}', pad=title_pad)
        plt.xlabel("Vocabulary Size", labelpad=label_pad)
        plt.ylabel(metric, labelpad=label_pad)
        plt.xticks(vocab_sizes, rotation=45)
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f'{figures_folder}/dice_domains.eps', bbox_inches='tight')
        # plt.savefig(f'{figures_folder}/dice_domains.pdf', bbox_inches='tight')
        # plt.savefig(f'{figures_folder}/dice_domains.png', bbox_inches='tight')
        plt.show()

def compare_tokenization_methods(df_train: pd.DataFrame, df_test: pd.DataFrame, tokenizer_cols: List[str], go_col: str, vocab_lineage_list: Dict,
                                 token_len_thr: int = 0, top_n_words: int = 10, raw_sequence_col: str = 'sequence', raw_or_tokenized: str = 'tokenized', 
                                 scoring_type: str = 'exist', lambda_smooth: float = 0.1, alpha: float = 1.0, beta: float = 0.5, theta: float = 0.7) -> pd.DataFrame:
    """
    Compares different tokenization methods by training and evaluating BERTopic models.

    Args:
        df_train: The training DataFrame.
        df_test: The testing DataFrame.
        tokenizer_cols: A list of tokenizer column names to compare.
        go_col: The name of the column with GO labels.
        vocab_lineage_list: A dictionary containing vocabulary lineage information.
        token_len_thr: The minimum token length.
        top_n_words: The number of words per topic.
        raw_sequence_col: The column with raw sequences.
        raw_or_tokenized: The evaluation mode ('raw' or 'tokenized').
        scoring_type: The scoring type ('exist' or 'freq').
        lambda_smooth, alpha, beta, theta: Parameters for the graph-aware model.

    Returns:
        A DataFrame summarizing the comparison results.
    """
    results = []

    for tokenizer_col in tqdm(tokenizer_cols, desc="Comparing Tokenizers"):
        print()
        print(f"--- Evaluating Tokenizer: {tokenizer_col} ---")
        documents_train = create_unit_documents(df_train, tokenizer_col, token_len_thr)
        go_labels_train = create_go_labels(df_train, go_col)

        # Standard BERTopic
        topic_model_std, _ = create_bertopic_model(documents_train, go_labels_train, token_len_thr, top_n_words)
        vocab_size = len(topic_model_std.get_params()['vectorizer_model'].vocabulary_)
        eval_std_single = evaluate_go_term_representations(df_test, topic_model_std, tokenizer_col, go_col, raw_sequence_col, raw_or_tokenized, scoring_type)
        eval_std_multi = evaluate_go_term_representations_multilabel(df_test, topic_model_std, tokenizer_col, go_col, raw_sequence_col, raw_or_tokenized, scoring_type)
        results.append({'tokenizer': tokenizer_col, 'model': 'Standard BERTopic', 'evaluation_single': eval_std_single, 'evaluation_multi': eval_std_multi, 'vocab_size': vocab_size})

        # Graph-Aware BERTopic
        unit_relationships = {'hierarchical': {}, 'mutational': {}}
        if tokenizer_col in vocab_lineage_list:
            for unit, lineage in vocab_lineage_list[tokenizer_col].items():
                if lineage.get('child_pair'): unit_relationships['hierarchical'][unit] = lineage['child_pair']
                if lineage.get('child_mutation'): unit_relationships['mutational'][unit] = lineage['child_mutation']
        
        topic_model_graph, _, _ = create_graph_aware_bertopic_model(documents_train, go_labels_train, unit_relationships, token_len_thr, top_n_words, lambda_smooth, alpha, beta, theta)
        vocab_size = len(topic_model_graph.get_params()['vectorizer_model'].vocabulary_)
        eval_graph_single = evaluate_go_term_representations(df_test, topic_model_graph, tokenizer_col, go_col, raw_sequence_col, raw_or_tokenized, scoring_type)
        eval_graph_multi = evaluate_go_term_representations_multilabel(df_test, topic_model_graph, tokenizer_col, go_col, raw_sequence_col, raw_or_tokenized, scoring_type)
        results.append({'tokenizer': tokenizer_col, 'model': 'Graph-Aware BERTopic', 'evaluation_single': eval_graph_single, 'evaluation_multi': eval_graph_multi, 'vocab_size': vocab_size})

    processed_results = []
    for res in results:
        single_summary = res['evaluation_single'].get('classification_summary', {})
        multi_summary = res['evaluation_multi'].get('classification_summary', {})
        processed_results.append({
            'Tokenizer': res['tokenizer'], 'Model': res['model'],
            'SL Accuracy': res['evaluation_single'].get('accuracy_at_k', {}).get('accuracy_at_1'),
            'SL Weighted Precision': single_summary.get('weighted_avg', {}).get('precision'),
            'SL Weighted Recall': single_summary.get('weighted_avg', {}).get('recall'),
            'SL Weighted F1': single_summary.get('weighted_avg', {}).get('f1-score'),
            'SL Accuracy at 5': res['evaluation_single'].get('accuracy_at_k', {}).get('accuracy_at_5'),
            'SL Precision at 5': res['evaluation_single'].get('precision_at_k', {}).get('precision_at_5'),
            'SL Recall at 5': res['evaluation_single'].get('recall_at_k', {}).get('recall_at_5'),
            'SL MRR': res['evaluation_single'].get('mean_reciprocal_rank'),
            'ML Accuracy': res['evaluation_multi'].get('accuracy_at_k', {}).get('accuracy_at_1'),
            'ML Weighted Precision': multi_summary.get('weighted_avg', {}).get('precision'),
            'ML Weighted Recall': multi_summary.get('weighted_avg', {}).get('recall'),
            'ML Weighted F1': multi_summary.get('weighted_avg', {}).get('f1-score'),
            'ML Accuracy at 5': res['evaluation_multi'].get('accuracy_at_k', {}).get('accuracy_at_5'),
            'ML Precision at 5': res['evaluation_multi'].get('precision_at_k', {}).get('precision_at_5'),
            'ML Recall at 5': res['evaluation_multi'].get('recall_at_k', {}).get('recall_at_5'),
            'ML MRR': res['evaluation_multi'].get('mean_reciprocal_rank'),
            'Vocab Size': res['vocab_size']
        })

    df_results = pd.DataFrame(processed_results)
    # print("\n--- Comparison Results ---")
    # print(df_results.to_string())
    return df_results

def optimize_graph_aware_parameters(df_train: pd.DataFrame, df_test: pd.DataFrame, tokenizer_col: str, go_col: str, vocab_lineage_list: Dict,
                                    raw_sequence_col: str, raw_or_tokenized: str, scoring_type: str,
                                    param_grid: dict, token_len_thr: int = 0, top_n_words: int = 10) -> Tuple[pd.DataFrame, Dict]:
    """
    Optimizes hyperparameters for the Graph-Aware BERTopic model.

    Args:
        df_train: The training DataFrame.
        df_test: The testing DataFrame.
        tokenizer_col: The tokenizer column to optimize for.
        go_col: The GO label column.
        vocab_lineage_list: A dictionary with vocabulary lineage information.
        raw_sequence_col: The column with raw sequences.
        raw_or_tokenized: The evaluation mode.
        scoring_type: The scoring type.
        param_grid: A dictionary of parameters to search.
        token_len_thr: The minimum token length.
        top_n_words: The number of words per topic.

    Returns:
        A tuple containing a DataFrame with optimization results and a dictionary of the best parameters.
    """
    results = []
    param_combinations = list(itertools.product(param_grid['lambda_smooth'], param_grid['alpha'], param_grid['beta'], param_grid['theta']))
    
    print(f"Testing {len(param_combinations)} parameter combinations for tokenizer: {tokenizer_col}")
    documents_train = create_unit_documents(df_train, tokenizer_col, token_len_thr)
    go_labels_train = create_go_labels(df_train, go_col)

    unit_relationships = {'hierarchical': {}, 'mutational': {}}
    if tokenizer_col in vocab_lineage_list:
        for unit, lineage in vocab_lineage_list[tokenizer_col].items():
            if lineage.get('child_pair'): unit_relationships['hierarchical'][unit] = lineage['child_pair']
            if lineage.get('child_mutation'): unit_relationships['mutational'][unit] = lineage['child_mutation']

    for lambda_smooth, alpha, beta, theta in tqdm(param_combinations, desc="Optimizing Parameters"):
        print()
        print(f"Testing lambda_smooth={lambda_smooth}, alpha={alpha}, beta={beta}, theta={theta}")
        topic_model_graph, _, _ = create_graph_aware_bertopic_model(documents_train, go_labels_train, unit_relationships, token_len_thr, top_n_words, lambda_smooth, alpha, beta, theta)
        eval_single = evaluate_go_term_representations(df_test, topic_model_graph, tokenizer_col, go_col, raw_sequence_col, raw_or_tokenized, scoring_type)
        eval_multi = evaluate_go_term_representations_multilabel(df_test, topic_model_graph, tokenizer_col, go_col, raw_sequence_col, raw_or_tokenized, scoring_type)

        single_summary = eval_single.get('classification_summary', {})
        multi_summary = eval_multi.get('classification_summary', {})
        results.append({
            'lambda_smooth': lambda_smooth, 'alpha': alpha, 'beta': beta, 'theta': theta,
            'SL Accuracy at 5': eval_single.get('accuracy_at_k', {}).get('accuracy_at_5'),
            'SL Weighted F1': single_summary.get('weighted_avg', {}).get('f1-score'),
            'SL MRR': eval_single.get('mean_reciprocal_rank'),
            'ML Accuracy at 5': eval_multi.get('accuracy_at_k', {}).get('accuracy_at_5'),
            'ML Weighted F1': multi_summary.get('weighted_avg', {}).get('f1-score'),
            'ML MRR': eval_multi.get('mean_reciprocal_rank'),
        })

    df_results = pd.DataFrame(results)
    
    best_params = {metric: df_results.loc[df_results[metric].idxmax()].iloc[:4].to_dict() for metric in list(df_results.columns[4:])}

    best_param_counter = {}
    for metric, params in best_params.items():
        for param, value in params.items():
            best_param_counter[f'{param}_{value}'] = best_param_counter.get(f'{param}_{value}', 0) + 1
    best_param_counter = sorted(best_param_counter.items(), key=lambda x: (x[0][0], x[1]), reverse=True)

    # print("\n--- Optimization Results ---")
    # print(df_results.to_string())
    # print("\n--- Best Parameters ---")
    # print(json.dumps(best_params, indent=2))
    print("\n--- Best Parameter Count ---")
    print(best_param_counter)

    return df_results, best_params, best_param_counter


class GOHierarchyComparator:
    def __init__(self, go_dag, goslim_terms, topic_model, documents):
        """
        Initialize the comparator with GO DAG and BERTopic data
        
        Args:
            go_dag: GODag object containing GO term relationships
            goslim_terms: Set of GO Slim terms
            topic_model: BERTopic model with custom labels as GO terms
            documents: Documents used in topic modeling
        """
        self.go_dag = go_dag
        self.goslim_terms = goslim_terms
        self.go_subdag_slim = GoSubDag(goslim_terms, go_dag)

        # Build BERTopic distance matrix
        distance_function = lambda x: ((1 - cosine_similarity(x))+((1 - cosine_similarity(x)).T))/2
        topic_distances = distance_function(topic_model.c_tf_idf_)
        self.bertopic_distance_matrix = topic_distances*(np.eye(topic_distances.shape[0])*-1+1)
        self.bertopic_distance_matrix = self._min_max_normalize(self.bertopic_distance_matrix)

        # Get BERTopic dendrogram
        self.bertopic_dendrogram = topic_model.hierarchical_topics(documents, distance_function=distance_function)

        # Build topic-GO mappings
        goslim_id2name = {go_id : go_dag[go_id].name for go_id in goslim_terms}
        goslim_name2id = {go_dag[go_id].name : go_id for go_id in goslim_terms}
        self.topic_to_go = {i:goslim_name2id[topic] for i, topic in enumerate(topic_model.custom_labels_)}
        self.go_to_topic = {goslim_name2id[topic]:i for i, topic in enumerate(topic_model.custom_labels_)}
        
        # Build GO network
        self.go_network = self._build_go_network()
                
        # Compute GO distance matrix (hop-based)
        self.go_distance_matrix = self._compute_go_distance_matrix()
        # self.go_distance_matrix_normalized = self._normalize_tensor(self.go_distance_matrix)
        self.go_distance_matrix_normalized = self._min_max_normalize(self.go_distance_matrix)

        # NEW: Convert BERTopic dendrogram to hop-based distance matrix
        self.bertopic_hop_distance_matrix = self._convert_dendrogram_to_hop_distances()

        # Build linkage matrices using distance matrices directly
        self.bertopic_linkage_matrix = self._build_linkage_matrix(
            self.bertopic_distance_matrix, self.bertopic_hop_distance_matrix, use_hop_distances=True, method='average'
        )
        self.go_linkage_matrix = self._build_linkage_matrix(
            self.go_distance_matrix_normalized, self.go_distance_matrix, use_hop_distances=True, method='average'
        )

    def _build_go_network(self):
        """Build NetworkX graph from GO DAG"""
        G = nx.DiGraph()
        for go_id, go_term in self.go_subdag_slim.go2obj.items():
            G.add_node(go_id, name=go_term.name, namespace=go_term.namespace)
            for parent in go_term.parents:
                if parent.id in self.go_subdag_slim.go2obj:  # Only connect if parent is also in slim
                    G.add_edge(parent.id, go_id)
        G.add_node('GO:0000000', name='root_node', namespace='root_node')
        G.add_edge('GO:0000000', 'GO:0008150') # biological_process
        G.add_edge('GO:0000000', 'GO:0005575') # cellular_component
        G.add_edge('GO:0000000', 'GO:0003674') # molecular_function

        return G
    
    def _compute_go_distance_matrix(self):
        """Compute hop-based distance matrix for GO terms"""
        go_terms = list(self.go_to_topic.keys())
        n_terms = len(go_terms)
        distance_matrix = np.zeros((n_terms, n_terms))
        
        # Convert to undirected for shortest path calculation
        G_undirected = self.go_network.to_undirected()
        
        for i, term1 in enumerate(go_terms):
            for j, term2 in enumerate(go_terms):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    try:
                        # Use shortest path length as hop distance
                        distance = nx.shortest_path_length(G_undirected, term1, term2)
                        distance_matrix[i, j] = distance
                    except nx.NetworkXNoPath:
                        # If no path exists, use maximum distance
                        distance_matrix[i, j] = n_terms * 2  # Large value for disconnected components
        
        return distance_matrix

    def _convert_dendrogram_to_hop_distances(self):
        """Convert BERTopic dendrogram to hop-based distance matrix"""
        n_topics = len(self.topic_to_go)
        hop_distance_matrix = np.zeros((n_topics, n_topics))
        
        # Create a mapping from topic pairs to their merge level (hop distance)
        topic_merge_levels = {}
        
        # Sort dendrogram by distance to process merges in order
        sorted_dendrogram = self.bertopic_dendrogram.sort_values('Distance').reset_index(drop=True)
        
        # Initialize each topic as its own cluster at level 0
        topic_clusters = {i: {i} for i in range(n_topics)}
        next_cluster_id = n_topics
        
        for merge_level, row in enumerate(sorted_dendrogram.iterrows()):
            _, row_data = row
            left_id = int(row_data['Child_Left_ID'])
            right_id = int(row_data['Child_Right_ID'])
            
            # Get topics in left and right clusters
            left_topics = topic_clusters.get(left_id, {left_id} if left_id < n_topics else set())
            right_topics = topic_clusters.get(right_id, {right_id} if right_id < n_topics else set())
            
            # Record hop distance between all pairs from different clusters
            for topic1 in left_topics:
                for topic2 in right_topics:
                    if topic1 < n_topics and topic2 < n_topics:
                        hop_distance = merge_level + 1  # +1 because we start from level 0
                        hop_distance_matrix[topic1, topic2] = hop_distance
                        hop_distance_matrix[topic2, topic1] = hop_distance
            
            # Create new cluster
            merged_topics = left_topics.union(right_topics)
            topic_clusters[next_cluster_id] = merged_topics
            next_cluster_id += 1
        
        return hop_distance_matrix

    def _build_linkage_matrix(self, cont_distance_matrix, hop_distance_matrix, use_hop_distances=True, method='average'):
        """
        Build linkage matrix from distance matrix
        
        Args:
            use_hop_distances: If True, use hop-based distances; if False, use cosine distances
            method: Linkage method ('average', 'complete', 'single', 'ward')
        """
        if use_hop_distances:
            # Use hop-based distances for structural comparison
            distance_matrix = hop_distance_matrix
            # Ward linkage is not appropriate for discrete hop distances
            if method == 'ward':
                print("Warning: Ward linkage not recommended for hop distances. Using 'average' instead.")
                method = 'average'
        else:
            # Use continuous cosine distances
            distance_matrix = cont_distance_matrix
        
        condensed_matrix = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_matrix, method=method)
        return linkage_matrix

    def _min_max_normalize(self, t):
        return (t - t.min()) / (t.max() - t.min())

    def _normalize_tensor(self, tensor, inf_replacement=2.0):
        """Normalize tensor with special handling for infinite/large values"""
        # Find the maximum finite value
        finite_mask = np.isfinite(tensor) & (tensor < tensor.max())
        if not np.any(finite_mask):
            return self._min_max_normalize(tensor)
        
        max_finite = tensor[finite_mask].max()
        print(tensor.max())
        # If all values are reasonably small, just normalize
        if tensor.max() < 20:
            return self._min_max_normalize(tensor)
        
        # Handle large values (disconnected components)
        inf_mask = tensor >= max_finite
        tensor_norm = tensor.copy()
        
        # Normalize finite values to [0, 1]
        finite_vals = tensor[~inf_mask]
        if len(finite_vals) > 0:
            finite_min, finite_max = finite_vals.min(), finite_vals.max()
            if finite_max > finite_min:
                tensor_norm[~inf_mask] = (finite_vals - finite_min) / (finite_max - finite_min)
            else:
                tensor_norm[~inf_mask] = 0
        
        # Replace large values with a constant > 1
        tensor_norm[inf_mask] = inf_replacement
        return tensor_norm

    def compare_linkage_approaches(self):
        """
        Compare different approaches to linkage calculation
        """
        approaches = {
            'hop_average': {'use_hop': True, 'method': 'average'},
            'hop_complete': {'use_hop': True, 'method': 'complete'},
            'continuous_ward': {'use_hop': False, 'method': 'ward'},
            'continuous_average': {'use_hop': False, 'method': 'average'}
        }
        
        results = {}
        
        for approach_name, params in approaches.items():
            # Build linkage matrices with specified parameters
            self.bertopic_linkage_matrix = self._build_linkage_matrix(
                self.bertopic_distance_matrix, self.bertopic_hop_distance_matrix,
                use_hop_distances=params['use_hop'], 
                method=params['method']
            )
            self.go_linkage_matrix = self._build_linkage_matrix(
                self.go_distance_matrix_normalized, self.go_distance_matrix,
                use_hop_distances=params['use_hop'], 
                method=params['method']
            )
            
            # Compute comparison metrics
            clustering_metrics = self.compare_hierarchical_clustering()
            cophenetic_metrics = self.compute_cophenetic_correlation()
            
            results[approach_name] = {
                'clustering_metrics': clustering_metrics,
                'cophenetic_correlation': cophenetic_metrics['cross_cophenetic_correlation'],
                'method': params['method'],
                'distance_type': 'hop-based' if params['use_hop'] else 'continuous'
            }
            
            print(f"\n{approach_name.upper()} Results:")
            print(f"  Method: {params['method']}, Distance: {'hop-based' if params['use_hop'] else 'continuous'}")
            print(f"  Cross-cophenetic correlation: {cophenetic_metrics['cross_cophenetic_correlation']:.4f}")
            avg_ari = np.mean([m['adjusted_rand_index'] for m in clustering_metrics.values()])
            print(f"  Average ARI: {avg_ari:.4f}")
        
        return results

    def compare_distance_matrices(self, use_hop_distances=True):
        """Compare BERTopic and GO distance matrices"""
        # Choose which BERTopic matrix to use
        if use_hop_distances:
            bert_matrix = self.bertopic_hop_distance_matrix
            go_matrix = self.go_distance_matrix
            comparison_type = "hop-based"
        else:
            bert_matrix = self.bertopic_distance_matrix
            go_matrix = self.go_distance_matrix_normalized
            comparison_type = "cosine-based"
        
        # Ensure matrices are same size
        min_size = min(bert_matrix.shape[0], go_matrix.shape[0])
        bert_matrix = bert_matrix[:min_size, :min_size]
        go_matrix = go_matrix[:min_size, :min_size]
        
        # Flatten upper triangular parts (excluding diagonal)
        bert_distances = bert_matrix[np.triu_indices_from(bert_matrix, k=1)]
        go_distances = go_matrix[np.triu_indices_from(go_matrix, k=1)]
        
        # Calculate correlations
        spearman_corr, spearman_p = spearmanr(bert_distances, go_distances)
        pearson_corr, pearson_p = pearsonr(bert_distances, go_distances)
        
        # Calculate normalized root mean square error
        rmse = np.sqrt(np.mean((bert_distances - go_distances) ** 2))
        normalized_rmse = rmse / np.std(go_distances) if np.std(go_distances) > 0 else float('inf')
        
        return {
            'comparison_type': comparison_type,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'rmse': rmse,
            'normalized_rmse': normalized_rmse
        }
    
    def compare_hierarchical_clustering(self, n_clusters_list=[3, 5, 7, 10]):
        """Compare hierarchical clustering results at different cut levels"""
        results = {}
        
        for n_clusters in n_clusters_list:
            # Get clustering from BERTopic linkage matrix
            bert_clusters = fcluster(self.bertopic_linkage_matrix, n_clusters, criterion='maxclust')
            
            # Get clustering from GO linkage matrix
            go_clusters = fcluster(self.go_linkage_matrix, n_clusters, criterion='maxclust')
            
            # Ensure same length
            min_len = min(len(bert_clusters), len(go_clusters))
            bert_clusters = bert_clusters[:min_len]
            go_clusters = go_clusters[:min_len]
            
            # Calculate similarity metrics
            if len(bert_clusters) == len(go_clusters) and len(set(bert_clusters)) > 1 and len(set(go_clusters)) > 1:
                ari = adjusted_rand_score(go_clusters, bert_clusters)
                nmi = normalized_mutual_info_score(go_clusters, bert_clusters)
                
                results[n_clusters] = {
                    'adjusted_rand_index': ari,
                    'normalized_mutual_info': nmi
                }
        
        return results
    
    def compute_cophenetic_correlation(self):
        """Compute cophenetic correlation coefficient for both hierarchies"""
        # BERTopic cophenetic correlation
        bert_condensed = squareform(self.bertopic_distance_matrix)
        bert_coph_corr, bert_coph_dists = cophenet(self.bertopic_linkage_matrix, bert_condensed)
        
        # GO cophenetic correlation
        go_condensed = squareform(self.go_distance_matrix_normalized)
        go_coph_corr, go_coph_dists = cophenet(self.go_linkage_matrix, go_condensed)
        
        # Correlation between cophenetic distance matrices
        min_len = min(len(bert_coph_dists), len(go_coph_dists))
        bert_coph_dists = bert_coph_dists[:min_len]
        go_coph_dists = go_coph_dists[:min_len]
        
        cross_coph_corr, cross_coph_p = pearsonr(bert_coph_dists, go_coph_dists)
        
        return {
            'bertopic_cophenetic_correlation': bert_coph_corr,
            'go_cophenetic_correlation': go_coph_corr,
            'cross_cophenetic_correlation': cross_coph_corr,
            'cross_cophenetic_p_value': cross_coph_p,
            'bertopic_cophenetic_distances': bert_coph_dists,
            'go_cophenetic_distances': go_coph_dists
        }
    
    def compute_fowlkes_mallows_index(self, n_clusters_list=[3, 5, 7, 10]):
        """Compute Fowlkes-Mallows Index at different clustering levels"""
        def fowlkes_mallows_score(labels_true, labels_pred):
            """Compute Fowlkes-Mallows Index manually"""
            labels_true = np.array(labels_true)
            labels_pred = np.array(labels_pred)
            
            n_samples = len(labels_true)
            
            # Compute contingency matrix
            contingency = {}
            for i in range(n_samples):
                true_label = labels_true[i]
                pred_label = labels_pred[i]
                if (true_label, pred_label) not in contingency:
                    contingency[(true_label, pred_label)] = 0
                contingency[(true_label, pred_label)] += 1
            
            # Compute TP, FP, FN
            tp = sum(count * (count - 1) // 2 for count in contingency.values())
            
            # Count pairs in same true cluster
            true_cluster_sizes = {}
            for label in labels_true:
                true_cluster_sizes[label] = true_cluster_sizes.get(label, 0) + 1
            tp_fp = sum(size * (size - 1) // 2 for size in true_cluster_sizes.values())
            
            # Count pairs in same predicted cluster
            pred_cluster_sizes = {}
            for label in labels_pred:
                pred_cluster_sizes[label] = pred_cluster_sizes.get(label, 0) + 1
            tp_fn = sum(size * (size - 1) // 2 for size in pred_cluster_sizes.values())
            
            if tp_fp == 0 or tp_fn == 0:
                return 0.0
            
            # Fowlkes-Mallows Index
            fmi = tp / np.sqrt(tp_fp * tp_fn)
            return fmi
        
        fmi_results = {}
        
        for n_clusters in n_clusters_list:
            # Get clustering from linkage matrices
            bert_clusters = fcluster(self.bertopic_linkage_matrix, n_clusters, criterion='maxclust')
            go_clusters = fcluster(self.go_linkage_matrix, n_clusters, criterion='maxclust')
            
            # Ensure same length
            min_len = min(len(bert_clusters), len(go_clusters))
            bert_clusters = bert_clusters[:min_len]
            go_clusters = go_clusters[:min_len]
            
            # Calculate Fowlkes-Mallows Index
            fmi = fowlkes_mallows_score(go_clusters, bert_clusters)
            
            fmi_results[n_clusters] = fmi
        
        return fmi_results
    
    def compare_subtree_structures(self, max_depth=3):
        """Compare local subtree structures between hierarchies"""
        # Extract subtree patterns from both hierarchies
        bert_subtree_patterns = self._extract_subtree_patterns(self.bertopic_linkage_matrix, max_depth)
        go_subtree_patterns = self._extract_subtree_patterns(self.go_linkage_matrix, max_depth)
        
        # Compare subtree patterns
        common_patterns = set(bert_subtree_patterns.keys()) & set(go_subtree_patterns.keys())
        bert_only_patterns = set(bert_subtree_patterns.keys()) - set(go_subtree_patterns.keys())
        go_only_patterns = set(go_subtree_patterns.keys()) - set(bert_subtree_patterns.keys())
        
        # Calculate pattern similarity
        total_bert_patterns = sum(bert_subtree_patterns.values())
        total_go_patterns = sum(go_subtree_patterns.values())
        common_pattern_score = sum(min(bert_subtree_patterns.get(p, 0), go_subtree_patterns.get(p, 0)) 
                                 for p in common_patterns)
        
        if total_bert_patterns + total_go_patterns > 0:
            jaccard_similarity = len(common_patterns) / (len(bert_subtree_patterns) + len(go_subtree_patterns) - len(common_patterns))
            pattern_overlap_score = 2 * common_pattern_score / (total_bert_patterns + total_go_patterns)
        else:
            jaccard_similarity = 0
            pattern_overlap_score = 0
        
        return {
            'jaccard_similarity': jaccard_similarity,
            'pattern_overlap_score': pattern_overlap_score,
            'num_common_patterns': len(common_patterns),
            'num_bert_only_patterns': len(bert_only_patterns),
            'num_go_only_patterns': len(go_only_patterns),
            'bert_subtree_patterns': bert_subtree_patterns,
            'go_subtree_patterns': go_subtree_patterns
        }
    
    def _extract_subtree_patterns(self, linkage_matrix, max_depth):
        """Extract subtree patterns from linkage matrix"""
        n_leaves = linkage_matrix.shape[0] + 1
        patterns = defaultdict(int)
        
        # Build tree structure from linkage matrix
        tree_structure = self._build_tree_from_linkage(linkage_matrix)
        
        # Extract patterns for each internal node
        for node_id in range(n_leaves, n_leaves + linkage_matrix.shape[0]):
            pattern = self._get_subtree_pattern(tree_structure, node_id, max_depth)
            if pattern:
                patterns[pattern] += 1
        
        return dict(patterns)
    
    def _build_tree_from_linkage(self, linkage_matrix):
        """Build tree structure from linkage matrix"""
        n_leaves = linkage_matrix.shape[0] + 1
        tree = {}
        
        for i, row in enumerate(linkage_matrix):
            node_id = n_leaves + i
            left_child = int(row[0])
            right_child = int(row[1])
            tree[node_id] = {'left': left_child, 'right': right_child}
        
        return tree
    
    def _get_subtree_pattern(self, tree, node_id, max_depth, current_depth=0):
        """Get subtree pattern as a string representation"""
        if current_depth >= max_depth or node_id not in tree:
            return "L"  # Leaf
        
        left_pattern = self._get_subtree_pattern(tree, tree[node_id]['left'], max_depth, current_depth + 1)
        right_pattern = self._get_subtree_pattern(tree, tree[node_id]['right'], max_depth, current_depth + 1)
        
        # Create a canonical representation
        if left_pattern <= right_pattern:
            return f"({left_pattern},{right_pattern})"
        else:
            return f"({right_pattern},{left_pattern})"
    
    def analyze_namespace_preservation(self):
        """Analyze how well BERTopic preserves GO namespace structure"""
        namespace_groups = defaultdict(list)
        for go_id in self.goslim_terms:
            if go_id in self.go_dag:
                namespace = self.go_dag[go_id].namespace
                topic_id = self.go_to_topic.get(go_id)
                if topic_id is not None:
                    namespace_groups[namespace].append(topic_id)
        
        # Use hop-based distances for better comparison
        within_namespace_distances = []
        between_namespace_distances = []
        
        for namespace, topics in namespace_groups.items():
            # Within-namespace distances
            for i, j in itertools.combinations(topics, 2):
                if i < self.bertopic_hop_distance_matrix.shape[0] and j < self.bertopic_hop_distance_matrix.shape[0]:
                    within_namespace_distances.append(self.bertopic_hop_distance_matrix[i, j])
        
        # Between-namespace distances
        namespaces = list(namespace_groups.keys())
        for ns1, ns2 in itertools.combinations(namespaces, 2):
            topics1 = namespace_groups[ns1]
            topics2 = namespace_groups[ns2]
            for i in topics1:
                for j in topics2:
                    if i < self.bertopic_hop_distance_matrix.shape[0] and j < self.bertopic_hop_distance_matrix.shape[0]:
                        between_namespace_distances.append(self.bertopic_hop_distance_matrix[i, j])
        
        # Calculate namespace preservation score
        if within_namespace_distances and between_namespace_distances:
            within_mean = np.mean(within_namespace_distances)
            between_mean = np.mean(between_namespace_distances)
            namespace_score = (between_mean - within_mean) / (between_mean + within_mean) if (between_mean + within_mean) > 0 else 0
        else:
            namespace_score = 0
        
        return {
            'namespace_preservation_score': namespace_score,
            'within_namespace_mean_distance': np.mean(within_namespace_distances) if within_namespace_distances else 0,
            'between_namespace_mean_distance': np.mean(between_namespace_distances) if between_namespace_distances else 0,
            'namespace_groups': dict(namespace_groups)
        }
    
    def analyze_parent_child_relationships(self):
        """Analyze how well BERTopic preserves parent-child relationships"""
        parent_child_bert_distances = []
        non_related_bert_distances = []
        
        # Get all parent-child pairs from GO network
        parent_child_pairs = list(self.go_network.edges())
        
        # Get all non-related pairs
        all_nodes = list(self.go_network.nodes())
        all_pairs = list(itertools.combinations(all_nodes, 2))
        non_related_pairs = [pair for pair in all_pairs if pair not in parent_child_pairs and (pair[1], pair[0]) not in parent_child_pairs]
        
        # Calculate distances for parent-child pairs using hop distances
        for parent, child in parent_child_pairs:
            parent_topic = self.go_to_topic.get(parent)
            child_topic = self.go_to_topic.get(child)
            if parent_topic is not None and child_topic is not None:
                if parent_topic < self.bertopic_hop_distance_matrix.shape[0] and child_topic < self.bertopic_hop_distance_matrix.shape[0]:
                    parent_child_bert_distances.append(self.bertopic_hop_distance_matrix[parent_topic, child_topic])
        
        # Calculate distances for non-related pairs (sample to avoid too many comparisons)
        non_related_sample = non_related_pairs[:len(parent_child_bert_distances) * 2]
        for node1, node2 in non_related_sample:
            topic1 = self.go_to_topic.get(node1)
            topic2 = self.go_to_topic.get(node2)
            if topic1 is not None and topic2 is not None:
                if topic1 < self.bertopic_hop_distance_matrix.shape[0] and topic2 < self.bertopic_hop_distance_matrix.shape[0]:
                    non_related_bert_distances.append(self.bertopic_hop_distance_matrix[topic1, topic2])
        
        # Calculate parent-child preservation score
        if parent_child_bert_distances and non_related_bert_distances:
            pc_mean = np.mean(parent_child_bert_distances)
            nr_mean = np.mean(non_related_bert_distances)
            pc_preservation_score = (nr_mean - pc_mean) / (nr_mean + pc_mean) if (nr_mean + pc_mean) > 0 else 0
        else:
            pc_preservation_score = 0
        
        return {
            'parent_child_preservation_score': pc_preservation_score,
            'parent_child_mean_distance': np.mean(parent_child_bert_distances) if parent_child_bert_distances else 0,
            'non_related_mean_distance': np.mean(non_related_bert_distances) if non_related_bert_distances else 0,
            'num_parent_child_pairs': len(parent_child_bert_distances),
            'num_non_related_pairs': len(non_related_bert_distances)
        }

    def analyze_dendrogram_structure(self):
        """Analyze the structural properties of both dendrograms"""
        # Calculate merge order correlation
        bert_merge_order = np.argsort(self.bertopic_dendrogram['Distance'].values)
        go_merge_order = np.argsort(self.go_linkage_matrix[:, 2])
        
        min_len = min(len(bert_merge_order), len(go_merge_order))
        merge_order_corr, merge_order_p = spearmanr(bert_merge_order[:min_len], go_merge_order[:min_len])
        
        # Calculate height variance similarity
        bert_height_var = np.var(self.bertopic_dendrogram['Distance'].values)
        go_height_var = np.var(self.go_linkage_matrix[:, 2])
        height_var_ratio = min(bert_height_var, go_height_var) / max(bert_height_var, go_height_var)
        
        return {
            'merge_order_correlation': merge_order_corr,
            'merge_order_p_value': merge_order_p,
            'height_variance_ratio': height_var_ratio,
            'bertopic_height_variance': bert_height_var,
            'go_height_variance': go_height_var
        }
    
    def _interpret_metric(self, metric_name, value):
        """Interpret metric values and provide qualitative assessment"""
        interpretations = {
            'spearman_correlation': {
                'range': (-1, 1),
                'thresholds': {'poor': 0.3, 'fair': 0.5, 'good': 0.7, 'excellent': 0.9},
                'higher_better': True,
                'description': 'Spearman correlation measures monotonic relationship between distance matrices'
            },
            'pearson_correlation': {
                'range': (-1, 1),
                'thresholds': {'poor': 0.3, 'fair': 0.5, 'good': 0.7, 'excellent': 0.9},
                'higher_better': True,
                'description': 'Pearson correlation measures linear relationship between distance matrices'
            },
            'normalized_rmse': {
                'range': (0, float('inf')),
                'thresholds': {'excellent': 0.2, 'good': 0.5, 'fair': 1.0, 'poor': float('inf')},
                'higher_better': False,
                'description': 'Normalized RMSE measures prediction error relative to data variability'
            },
            'adjusted_rand_index': {
                'range': (-1, 1),
                'thresholds': {'poor': 0.2, 'fair': 0.4, 'good': 0.6, 'excellent': 0.8},
                'higher_better': True,
                'description': 'ARI measures similarity between clusterings, adjusted for chance'
            },
            'normalized_mutual_info': {
                'range': (0, 1),
                'thresholds': {'poor': 0.2, 'fair': 0.4, 'good': 0.6, 'excellent': 0.8},
                'higher_better': True,
                'description': 'NMI measures mutual information between clusterings, normalized'
            },
            'fowlkes_mallows_index': {
                'range': (0, 1),
                'thresholds': {'poor': 0.3, 'fair': 0.5, 'good': 0.7, 'excellent': 0.9},
                'higher_better': True,
                'description': 'FMI measures similarity between clusterings using precision and recall'
            },
            'cophenetic_correlation': {
                'range': (0, 1),
                'thresholds': {'poor': 0.5, 'fair': 0.7, 'good': 0.8, 'excellent': 0.9},
                'higher_better': True,
                'description': 'Cophenetic correlation measures how well the hierarchy preserves original distances'
            },
            'namespace_preservation_score': {
                'range': (-1, 1),
                'thresholds': {'poor': 0.1, 'fair': 0.3, 'good': 0.5, 'excellent': 0.7},
                'higher_better': True,
                'description': 'Measures how well GO namespace structure is preserved in BERTopic hierarchy'
            },
            'parent_child_preservation_score': {
                'range': (-1, 1),
                'thresholds': {'poor': 0.1, 'fair': 0.3, 'good': 0.5, 'excellent': 0.7},
                'higher_better': True,
                'description': 'Measures how well GO parent-child relationships are preserved'
            },
            'jaccard_similarity': {
                'range': (0, 1),
                'thresholds': {'poor': 0.2, 'fair': 0.4, 'good': 0.6, 'excellent': 0.8},
                'higher_better': True,
                'description': 'Jaccard similarity between subtree pattern sets'
            },
            'pattern_overlap_score': {
                'range': (0, 1),
                'thresholds': {'poor': 0.2, 'fair': 0.4, 'good': 0.6, 'excellent': 0.8},
                'higher_better': True,
                'description': 'Overlap score for subtree patterns considering frequencies'
            }
        }
        
        if metric_name not in interpretations:
            return "Unknown metric"
        
        info = interpretations[metric_name]
        higher_better = info['higher_better']
        thresholds = info['thresholds']
        
        if higher_better:
            if value >= thresholds['excellent']:
                quality = "excellent"
            elif value >= thresholds['good']:
                quality = "good"
            elif value >= thresholds['fair']:
                quality = "fair"
            else:
                quality = "poor"
        else:
            if value <= thresholds['excellent']:
                quality = "excellent"
            elif value <= thresholds['good']:
                quality = "good"
            elif value <= thresholds['fair']:
                quality = "fair"
            else:
                quality = "poor"
        
        return {
            'quality': quality,
            'description': info['description'],
            'range': info['range'],
            'value': value
        }

    def compute_comprehensive_score(self):
        """Compute a comprehensive comparison score"""
        # Get individual metrics
        distance_metrics_cosine = self.compare_distance_matrices(use_hop_distances=False)
        distance_metrics_hop = self.compare_distance_matrices(use_hop_distances=True)
        clustering_metrics = self.compare_hierarchical_clustering()
        namespace_metrics = self.analyze_namespace_preservation()
        parent_child_metrics = self.analyze_parent_child_relationships()
        
        # Dendrogram-based metrics
        cophenetic_metrics = self.compute_cophenetic_correlation()
        fmi_metrics = self.compute_fowlkes_mallows_index()
        dendrogram_structure_metrics = self.analyze_dendrogram_structure()
        subtree_metrics = self.compare_subtree_structures()
        
        # Weighted combination of metrics (focusing on hop-based distances)
        score_components = {
            'hop_distance_correlation': distance_metrics_hop['spearman_correlation'] * 0.20,
            'cosine_distance_correlation': distance_metrics_cosine['spearman_correlation'] * 0.10,
            'clustering_similarity': np.mean([metrics['adjusted_rand_index'] for metrics in clustering_metrics.values()]) * 0.15,
            'namespace_preservation': max(0, namespace_metrics['namespace_preservation_score']) * 0.10,
            'parent_child_preservation': max(0, parent_child_metrics['parent_child_preservation_score']) * 0.10,
            'cophenetic_correlation': cophenetic_metrics['cross_cophenetic_correlation'] * 0.15,
            'fowlkes_mallows_index': np.mean(list(fmi_metrics.values())) * 0.10,
            'dendrogram_structure': dendrogram_structure_metrics['merge_order_correlation'] * 0.05,
            'subtree_similarity': subtree_metrics['jaccard_similarity'] * 0.05
        }
        
        comprehensive_score = sum(score_components.values())
        
        return {
            'comprehensive_score': comprehensive_score,
            'score_components': score_components,
            'detailed_metrics': {
                'distance_metrics_cosine': distance_metrics_cosine,
                'distance_metrics_hop': distance_metrics_hop,
                'clustering_metrics': clustering_metrics,
                'namespace_metrics': namespace_metrics,
                'parent_child_metrics': parent_child_metrics,
                'cophenetic_metrics': cophenetic_metrics,
                'fowlkes_mallows_metrics': fmi_metrics,
                'dendrogram_structure_metrics': dendrogram_structure_metrics,
                'subtree_metrics': subtree_metrics
            }
        }
    
    def visualize_comparison(self, save_path=None):
        """Create visualizations comparing the hierarchies"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 24))
        
        # Plot 1: Hop-based distance matrix comparison
        ax1 = axes[0, 0]
        min_size = min(self.bertopic_hop_distance_matrix.shape[0], self.go_distance_matrix.shape[0])
        bert_hop_matrix = self.bertopic_hop_distance_matrix[:min_size, :min_size]
        go_matrix = self.go_distance_matrix[:min_size, :min_size]
        
        bert_hop_distances = bert_hop_matrix[np.triu_indices_from(bert_hop_matrix, k=1)]
        go_distances = go_matrix[np.triu_indices_from(go_matrix, k=1)]
        
        ax1.scatter(go_distances, bert_hop_distances, alpha=0.6, color='blue')
        ax1.set_xlabel('GO Hop Distance')
        ax1.set_ylabel('BERTopic Hop Distance')
        ax1.set_title('Hop-based Distance Matrix Comparison')
        
        # Add correlation info
        hop_corr = self.compare_distance_matrices(use_hop_distances=True)
        ax1.text(0.05, 0.95, f"Spearman r = {hop_corr['spearman_correlation']:.3f}", 
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Cosine-based distance matrix comparison
        ax2 = axes[0, 1]
        bert_cosine_matrix = self.bertopic_distance_matrix[:min_size, :min_size]
        bert_cosine_distances = bert_cosine_matrix[np.triu_indices_from(bert_cosine_matrix, k=1)]

        go_cosine_matrix = self.go_distance_matrix_normalized[:min_size, :min_size]
        go_cosine_distances = go_cosine_matrix[np.triu_indices_from(go_cosine_matrix, k=1)]
        
        ax2.scatter(go_cosine_distances, bert_cosine_distances, alpha=0.6, color='red')
        ax2.set_xlabel('GO Cosine Distance')
        ax2.set_ylabel('BERTopic Cosine Distance')
        ax2.set_title('Cosine-based Distance Matrix Comparison')
        
        # Add correlation info
        cosine_corr = self.compare_distance_matrices(use_hop_distances=False)
        ax2.text(0.05, 0.95, f"Spearman r = {cosine_corr['spearman_correlation']:.3f}", 
                transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Cophenetic distance comparison
        ax3 = axes[1, 0]
        cophenetic_metrics = self.compute_cophenetic_correlation()
        bert_coph = cophenetic_metrics['bertopic_cophenetic_distances']
        go_coph = cophenetic_metrics['go_cophenetic_distances']
        
        ax3.scatter(go_coph, bert_coph, alpha=0.6, color='orange')
        ax3.set_xlabel('GO Cophenetic Distance')
        ax3.set_ylabel('BERTopic Cophenetic Distance')
        ax3.set_title(f'Cophenetic Distance Comparison\nr = {cophenetic_metrics["cross_cophenetic_correlation"]:.3f}')
        
        # Plot 4: BERTopic hop distance matrix heatmap
        ax4 = axes[1, 1]
        sns.heatmap(bert_hop_matrix, ax=ax4, cmap='viridis', cbar=True, square=True)
        ax4.set_title('BERTopic Hop Distance Matrix')
        
        # Plot 5: GO distance matrix heatmap
        ax5 = axes[2, 0]
        sns.heatmap(go_matrix, ax=ax5, cmap='viridis', cbar=True, square=True)
        ax5.set_title('GO Hop Distance Matrix')
        
        # Plot 6: Clustering comparison metrics
        ax6 = axes[2, 1]
        clustering_results = self.compare_hierarchical_clustering()
        fmi_results = self.compute_fowlkes_mallows_index()
        
        n_clusters = list(clustering_results.keys())
        ari_scores = [clustering_results[k]['adjusted_rand_index'] for k in n_clusters]
        nmi_scores = [clustering_results[k]['normalized_mutual_info'] for k in n_clusters]
        fmi_scores = [fmi_results[k] for k in n_clusters]
        
        ax6.plot(n_clusters, ari_scores, 'o-', label='Adjusted Rand Index', linewidth=2)
        ax6.plot(n_clusters, nmi_scores, 's-', label='Normalized Mutual Info', linewidth=2)
        ax6.plot(n_clusters, fmi_scores, '^-', label='Fowlkes-Mallows Index', linewidth=2)
        ax6.set_xlabel('Number of Clusters')
        ax6.set_ylabel('Similarity Score')
        ax6.set_title('Clustering Similarity vs Number of Clusters')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        # Plot 7: Namespace preservation analysis
        ax7 = axes[3, 0]
        namespace_metrics = self.analyze_namespace_preservation()
        
        # Create bar plot for namespace preservation
        within_dist = namespace_metrics['within_namespace_mean_distance']
        between_dist = namespace_metrics['between_namespace_mean_distance']
        
        bars = ax7.bar(['Within Namespace', 'Between Namespace'], [within_dist, between_dist], 
                      color=['green', 'red'], alpha=0.7)
        ax7.set_ylabel('Mean Distance')
        ax7.set_title(f'Namespace Preservation\nScore: {namespace_metrics["namespace_preservation_score"]:.3f}')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 8: Parent-child relationship preservation
        ax8 = axes[3, 1]
        pc_metrics = self.analyze_parent_child_relationships()
        
        # Create bar plot for parent-child preservation
        pc_dist = pc_metrics['parent_child_mean_distance']
        nr_dist = pc_metrics['non_related_mean_distance']
        
        bars = ax8.bar(['Parent-Child', 'Non-Related'], [pc_dist, nr_dist], 
                      color=['blue', 'orange'], alpha=0.7)
        ax8.set_ylabel('Mean Distance')
        ax8.set_title(f'Parent-Child Preservation\nScore: {pc_metrics["parent_child_preservation_score"]:.3f}')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig

# Updated comparison function
def compare_hierarchies(go_dag, goslim_terms, topic_model, documents):
    """
    Main function to compare GO and BERTopic hierarchies
    
    Args:
        go_dag: GODag object
        goslim_terms: Set of GO Slim terms
        topic_model: BERTopic model with custom labels as GO terms
        documents: Documents used in topic modeling
    
    Returns:
        Dictionary with comprehensive comparison results
    """
    comparator = GOHierarchyComparator(go_dag, goslim_terms, topic_model, documents)
    
    # Compute comprehensive comparison
    results = comparator.compute_comprehensive_score()
    
    # Create visualizations
    comparator.visualize_comparison()
    
    # Print summary with interpretations
    print("=== GO Hierarchy Comparison Results ===")
    print(f"Comprehensive Score: {results['comprehensive_score']:.4f}")
    print(f"Score Range: [-1.0, 1.0] (higher is better)")
    
    print("\n=== Score Components ===")
    for component, score in results['score_components'].items():
        print(f"  {component}: {score:.4f}")
    
    print("\n=== Detailed Metrics with Interpretations ===")
    
    # Distance metrics
    print("\n--- Distance Correlation Metrics ---")
    hop_metrics = results['detailed_metrics']['distance_metrics_hop']
    cosine_metrics = results['detailed_metrics']['distance_metrics_cosine']
    
    hop_interp = comparator._interpret_metric('spearman_correlation', hop_metrics['spearman_correlation'])
    cosine_interp = comparator._interpret_metric('spearman_correlation', cosine_metrics['spearman_correlation'])
    
    print(f"  Hop-based Spearman Correlation: {hop_metrics['spearman_correlation']:.4f} ({hop_interp['quality']})")
    print(f"    Range: {hop_interp['range']}, {hop_interp['description']}")
    print(f"  Cosine-based Spearman Correlation: {cosine_metrics['spearman_correlation']:.4f} ({cosine_interp['quality']})")
    print(f"    Range: {cosine_interp['range']}, {cosine_interp['description']}")
    
    # Clustering metrics
    print("\n--- Clustering Similarity Metrics ---")
    clustering_metrics = results['detailed_metrics']['clustering_metrics']
    fmi_metrics = results['detailed_metrics']['fowlkes_mallows_metrics']
    
    avg_ari = np.mean([metrics['adjusted_rand_index'] for metrics in clustering_metrics.values()])
    avg_nmi = np.mean([metrics['normalized_mutual_info'] for metrics in clustering_metrics.values()])
    avg_fmi = np.mean(list(fmi_metrics.values()))
    
    ari_interp = comparator._interpret_metric('adjusted_rand_index', avg_ari)
    nmi_interp = comparator._interpret_metric('normalized_mutual_info', avg_nmi)
    fmi_interp = comparator._interpret_metric('fowlkes_mallows_index', avg_fmi)
    
    print(f"  Average Adjusted Rand Index: {avg_ari:.4f} ({ari_interp['quality']})")
    print(f"    Range: {ari_interp['range']}, {ari_interp['description']}")
    print(f"  Average Normalized Mutual Info: {avg_nmi:.4f} ({nmi_interp['quality']})")
    print(f"    Range: {nmi_interp['range']}, {nmi_interp['description']}")
    print(f"  Average Fowlkes-Mallows Index: {avg_fmi:.4f} ({fmi_interp['quality']})")
    print(f"    Range: {fmi_interp['range']}, {fmi_interp['description']}")
    
    # Cophenetic correlation
    print("\n--- Cophenetic Correlation ---")
    cophenetic_metrics = results['detailed_metrics']['cophenetic_metrics']
    coph_interp = comparator._interpret_metric('cophenetic_correlation', cophenetic_metrics['cross_cophenetic_correlation'])
    
    print(f"  Cross-Cophenetic Correlation: {cophenetic_metrics['cross_cophenetic_correlation']:.4f} ({coph_interp['quality']})")
    print(f"    Range: {coph_interp['range']}, {coph_interp['description']}")
    
    # Dendrogram structure
    print("\n--- Dendrogram Structure ---")
    dendrogram_metrics = results['detailed_metrics']['dendrogram_structure_metrics']
    merge_interp = comparator._interpret_metric('spearman_correlation', dendrogram_metrics['merge_order_correlation'])
    
    print(f"  Merge Order Correlation: {dendrogram_metrics['merge_order_correlation']:.4f} ({merge_interp['quality']})")
    print(f"    Range: {merge_interp['range']}, {merge_interp['description']}")
    
    # GO-specific metrics
    print("\n--- GO-Specific Preservation ---")
    namespace_metrics = results['detailed_metrics']['namespace_metrics']
    parent_child_metrics = results['detailed_metrics']['parent_child_metrics']
    
    ns_interp = comparator._interpret_metric('namespace_preservation_score', namespace_metrics['namespace_preservation_score'])
    pc_interp = comparator._interpret_metric('parent_child_preservation_score', parent_child_metrics['parent_child_preservation_score'])
    
    print(f"  Namespace Preservation: {namespace_metrics['namespace_preservation_score']:.4f} ({ns_interp['quality']})")
    print(f"    Range: {ns_interp['range']}, {ns_interp['description']}")
    print(f"  Parent-Child Preservation: {parent_child_metrics['parent_child_preservation_score']:.4f} ({pc_interp['quality']})")
    print(f"    Range: {pc_interp['range']}, {pc_interp['description']}")
    
    # Subtree structure
    print("\n--- Subtree Structure ---")
    subtree_metrics = results['detailed_metrics']['subtree_metrics']
    jaccard_interp = comparator._interpret_metric('jaccard_similarity', subtree_metrics['jaccard_similarity'])
    pattern_interp = comparator._interpret_metric('pattern_overlap_score', subtree_metrics['pattern_overlap_score'])
    
    print(f"  Jaccard Similarity: {subtree_metrics['jaccard_similarity']:.4f} ({jaccard_interp['quality']})")
    print(f"    Range: {jaccard_interp['range']}, {jaccard_interp['description']}")
    print(f"  Pattern Overlap Score: {subtree_metrics['pattern_overlap_score']:.4f} ({pattern_interp['quality']})")
    print(f"    Range: {pattern_interp['range']}, {pattern_interp['description']}")
    
    print(f"\n=== Summary ===")
    print(f"Total patterns compared: {subtree_metrics['num_common_patterns']} common, "
          f"{subtree_metrics['num_bert_only_patterns']} BERTopic-only, "
          f"{subtree_metrics['num_go_only_patterns']} GO-only")
    
    return results

def compare_tokenization_methods_hierarchical(df_train: pd.DataFrame, tokenizer_cols: List[str], go_col: str, go_dag: any, goslim_terms: Set[str], vocab_lineage_list: Dict,
                                              token_len_thr: int = 0, top_n_words: int = 10, lambda_smooth: float = 0.1, alpha: float = 1.0, beta: float = 0.5, theta: float = 0.7) -> pd.DataFrame:
    """
    Compares different tokenization methods by training BERTopic models and evaluating their hierarchical congruence with the GO DAG.

    Args:
        df_train: The training DataFrame.
        tokenizer_cols: A list of tokenizer column names to compare.
        go_col: The name of the column with GO labels.
        go_dag: The GODag object.
        goslim_terms: A set of GO Slim terms.
        vocab_lineage_list: A dictionary containing vocabulary lineage information.
        token_len_thr: The minimum token length.
        top_n_words: The number of words per topic.
        lambda_smooth, alpha, beta, theta: Parameters for the graph-aware model.

    Returns:
        A DataFrame summarizing the hierarchical comparison results.
    """
    results = []

    for tokenizer_col in tqdm(tokenizer_cols, desc="Comparing Tokenizers for Hierarchy"):
        print()
        print(f"--- Evaluating Hierarchy for Tokenizer: {tokenizer_col} ---")
        documents_train = create_unit_documents(df_train, tokenizer_col, token_len_thr)
        go_labels_train = create_go_labels(df_train, go_col)

        # Standard BERTopic
        topic_model_std, _ = create_bertopic_model(documents_train, go_labels_train, token_len_thr, top_n_words)
        comparator_std = GOHierarchyComparator(go_dag, goslim_terms, topic_model_std, documents_train)
        eval_std = comparator_std.compute_comprehensive_score()
        results.append({
            'tokenizer': tokenizer_col,
            'model': 'Standard BERTopic',
            'comprehensive_score': eval_std['comprehensive_score'],
            **eval_std['score_components']
        })

        # Graph-Aware BERTopic
        unit_relationships = {'hierarchical': {}, 'mutational': {}}
        if tokenizer_col in vocab_lineage_list:
            for unit, lineage in vocab_lineage_list[tokenizer_col].items():
                if lineage.get('child_pair'):
                    unit_relationships['hierarchical'][unit] = lineage['child_pair']
                if lineage.get('child_mutation'):
                    unit_relationships['mutational'][unit] = lineage['child_mutation']
        
        topic_model_graph, _, _ = create_graph_aware_bertopic_model(
            documents_train, go_labels_train, unit_relationships, token_len_thr, top_n_words, lambda_smooth, alpha, beta, theta
        )
        comparator_graph = GOHierarchyComparator(go_dag, goslim_terms, topic_model_graph, documents_train)
        eval_graph = comparator_graph.compute_comprehensive_score()
        results.append({
            'tokenizer': tokenizer_col,
            'model': 'Graph-Aware BERTopic',
            'comprehensive_score': eval_graph['comprehensive_score'],
            **eval_graph['score_components']
        })
        
    return pd.DataFrame(results)