import pandas as pd
import numpy as np
import ast
from collections import defaultdict
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

#Get Ground Truth Experts (filter authors with tag == query and rank by citations)
def get_ground_truth_experts(authors_df, query_tag, top_k):
    authors_df = authors_df.copy()
    authors_df["id"] = authors_df["id"].astype(int)  # match with expert_ids type 

    # Filter out rows with NaN tags
    valid_df = authors_df[authors_df['tags'].notna()]

    # Function to check if tag list contains  query
    def match_tag(tag_str):
        try:
            tag_list = ast.literal_eval(tag_str)  # Convert string to list-of-dicts
            return any(tag_obj.get('t', '').lower() == query_tag.lower() for tag_obj in tag_list)
        except:
            return False  

    # Filter authors matching the query tag
    matched_df = valid_df[valid_df['tags'].apply(match_tag)]

    # Sort by n_citation (descending)
    ranked_df = matched_df.sort_values(by='n_citation', ascending=False)

    # Get top_k author ids
    ground_truth_ids = ranked_df['id'].tolist()#[:top_k]

    return ground_truth_ids


# Precision@K (same as MP@K because it’s mean precision at cutoff K)
def mean_precision_at_k(predicted_ids, ground_truth_ids, k):
    predicted_at_k = predicted_ids[:k]
    hits = [1 if pid in ground_truth_ids else 0 for pid in predicted_at_k]
    return np.mean(hits) if hits else 0.0

# Mean Average Precision (MAP@K)
def average_precision(predicted_ids, ground_truth_ids, k):
    hits = 0
    sum_precisions = 0
    for i, pid in enumerate(predicted_ids[:k]):
        if pid in ground_truth_ids:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / hits if hits > 0 else 0.0

# MRR@K
def reciprocal_rank(predicted_ids, ground_truth_ids, k):
    for i, pid in enumerate(predicted_ids[:k]):
        if pid in ground_truth_ids:
            return 1 / (i + 1)
    return 0.0

# NDCG@K
def ndcg_at_k(predicted_ids, ground_truth_ids, k):
    dcg = 0
    for i, pid in enumerate(predicted_ids[:k]):
        rel = 1 if pid in ground_truth_ids else 0
        dcg += (2**rel - 1) / np.log2(i + 2)

    # Ideal DCG (best ranking)
    ideal_rels = [1] * min(len(ground_truth_ids), k)
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


# Global Recall (no K cutoff)
def recall(predicted_ids, ground_truth_ids):
    hits = sum(1 for gid in ground_truth_ids if gid in predicted_ids)
    return hits / len(ground_truth_ids) if ground_truth_ids else 0.0


# Full evaluation
def evaluate_expert_ranking(predicted_ids, ground_truth_ids):
    results = {
        'MAP@10': average_precision(predicted_ids, ground_truth_ids, 10),
        'MRR@10': reciprocal_rank(predicted_ids, ground_truth_ids, 10),
        'MP@5': mean_precision_at_k(predicted_ids, ground_truth_ids, 5),  # same as Precision@5
        'MP@10': mean_precision_at_k(predicted_ids, ground_truth_ids, 10), # same as Precision@10
        'NDCG@5': ndcg_at_k(predicted_ids, ground_truth_ids, 5),
        'NDCG@10': ndcg_at_k(predicted_ids, ground_truth_ids, 10)
        # 'Recall': recall(predicted_ids, ground_truth_ids)
    }
    return results






