import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import itertools
import json
import random
from collections import Counter

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

import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Dict, Tuple, Set, Any, Optional, Union

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
def create_bertopic_model(documents: List[str], go_labels: List[str], token_len_thr: int = 0) -> Tuple[BERTopic, np.ndarray]:
    """
    Creates and fits a standard BERTopic model for manual topic modeling based on GO labels.

    Args:
        documents: A list of protein sequences represented as documents.
        go_labels: A list of GO labels corresponding to each document.
        token_len_thr: The minimum length of a token to be included.

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
    
    # Match topic names with GO term IDs
    topic_labels = {i: lb.classes_[i] for i in range(len(lb.classes_))}
    topic_labels[-1] = "Outlier"
    
    # Fit the model with the documents and manually assigned topics
    topic_model.fit_transform(documents, y=topics)
    topic_model.set_topic_labels(topic_labels)
    
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
        lambda_smooth: The graph smoothing parameter.
        alpha: Weight for hierarchical relationships.
        beta: Weight for sibling relationships.
        theta: Weight for mutational relationships.

    Returns:
        A tuple containing the trained model, topics array, and the similarity matrix.
    """
    
    lb = LabelBinarizer()
    go_binary = lb.fit_transform(go_labels)
    
    temp_topic_model, _ = create_bertopic_model(documents, go_labels, token_len_thr)
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
        verbose=False
    )
    
    topics = np.argmax(go_binary, axis=1)
    topics[np.sum(go_binary, axis=1) == 0] = -1
    
    topic_labels = {i: lb.classes_[i] for i in range(len(lb.classes_))}
    topic_labels[-1] = "Outlier"
    
    topic_model.fit_transform(documents, y=topics)
    topic_model.set_topic_labels(topic_labels)
    
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
                                 token_len_thr: int = 0, raw_sequence_col: str = 'sequence', raw_or_tokenized: str = 'tokenized', 
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
        topic_model_std, _ = create_bertopic_model(documents_train, go_labels_train, token_len_thr)
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
        
        topic_model_graph, _, _ = create_graph_aware_bertopic_model(documents_train, go_labels_train, unit_relationships, token_len_thr, lambda_smooth, alpha, beta, theta)
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
                                    param_grid: dict, token_len_thr: int = 0) -> Tuple[pd.DataFrame, Dict]:
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
        topic_model_graph, _, _ = create_graph_aware_bertopic_model(documents_train, go_labels_train, unit_relationships, token_len_thr, lambda_smooth, alpha, beta, theta)
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
