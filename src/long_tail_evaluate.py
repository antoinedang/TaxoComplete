import argparse
import torch
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


torch.manual_seed(0)
args = argparse.ArgumentParser(description="Run error analysis on evaluation results.")
args.add_argument(
    type=str,
    help="Error analysis pickle file path",
    dest="filename",
)
args = args.parse_args()

error_analysis_dir = os.path.dirname(args.filename)
plot_filename = error_analysis_dir + "/embedding_cosine_similarities.png"

with open(args.filename, "rb") as f:
    (
        taxonomy,
        data_prep,
        query_embeddings,
        targets,
        all_predictions,
        all_predictions_ppr,
        edges_predictions_test,
        edges_predictions_test_ppr,
        corpus_embeddings,
        nodeId2corpusId,
        preds,
    ) = CPU_Unpickler(f).load()

core_subgraph_undirected = data_prep.core_subgraph.to_undirected()
core_subgraph_undirected.remove_node(data_prep.pseudo_leaf_node)


def get_embedding(node_id):
    if node_id == data_prep.pseudo_leaf_node:
        return None
    else:
        return corpus_embeddings[nodeId2corpusId[node_id]]


def get_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return "N/A"
    else:
        return torch.cosine_similarity(embedding1, embedding2, dim=0)

if not os.path.exists(error_analysis_dir + "/cosine_similarities.pkl"):
    cosine_similarities = []

    #   FOR EACH QUERY:
    try:
        for i in range(len(data_prep.test_queries)):
            print(f"Query {i+1}/{len(data_prep.test_queries)}")
            query_embedding = query_embeddings[i]
            for corpus_embedding in corpus_embeddings:
                cos_sim = get_cosine_similarity(query_embedding, corpus_embedding)
                if cos_sim != "N/A":
                    cosine_similarities.append(float(cos_sim))
    except Exception as e:
        print(e)

    with open(error_analysis_dir + "/cosine_similarities.pkl", "wb") as f:
        pickle.dump(cosine_similarities, f)
else:
    with open(error_analysis_dir + "/cosine_similarities.pkl", "rb") as f:
        cosine_similarities = pickle.load(f)

def is_correct_prediction(edges, target, checkChild, K):
    idx = 1 if checkChild else 0
    return any(
        [
            any(
                [
                    edges[i][n][idx] == sub_target[idx]
                    or (
                        edges[i][n][idx] == data_prep.pseudo_leaf_node
                        and not (
                            sub_target[idx] in list(data_prep.core_subgraph.nodes())
                        )
                    )
                    for sub_target in target
                ]
            )
            for n in range(K)
        ]
    )
    
def rank_of_correct_prediction(edges, target, checkChild):
    rank = 0
    found_correct = False
    while not found_correct:
        rank += 1
        found_correct = is_correct_prediction(edges, target, checkChild, rank)
    return rank
    
hit_at_1_below_80th = []
hit_at_1_above_80th = []
hit_at_5_below_80th = []
hit_at_5_above_80th = []
hit_at_10_below_80th = []
hit_at_10_above_80th = []
MR_below_80th = []
MR_above_80th = []

cosine_similarities_sorted = np.sort(cosine_similarities)
percentile_80 = np.percentile(cosine_similarities_sorted, 80)

#   FOR EACH QUERY:
for i in range(len(data_prep.test_queries)):
    query = data_prep.test_queries[i]
    target = targets[i]
    
    # RELEVANCE: isCorrectParent, isCorrectChild, isCorrectParentPPR, isCorrectChildPPR
    isCorrectParentPPRAt1 = is_correct_prediction(
        edges_predictions_test_ppr, target, checkChild=False, K=1
    )
    isCorrectParentPPRAt5 = is_correct_prediction(
        edges_predictions_test_ppr, target, checkChild=False, K=5
    )
    isCorrectParentPPRAt10 = is_correct_prediction(
        edges_predictions_test_ppr, target, checkChild=False, K=10
    )
    isCorrectChildPPRAt1 = is_correct_prediction(
        edges_predictions_test_ppr, target, checkChild=True, K=1
    )
    isCorrectChildPPRAt5 = is_correct_prediction(
        edges_predictions_test_ppr, target, checkChild=True, K=5
    )
    isCorrectChildPPRAt10 = is_correct_prediction(
        edges_predictions_test_ppr, target, checkChild=True, K=10
    )
    rank_child = rank_of_correct_prediction(edges_predictions_test_ppr, target, checkChild=True)
    rank_parent = rank_of_correct_prediction(edges_predictions_test_ppr, target, checkChild=False)
    for sub_target in target:
        # COSINE SIMILARITY: cos_similarity_untrained(query node, actual parent), cos_similarity_untrained(query node, actual child)
        index_query = data_prep.test_queries.index(query)
        true_parent = sub_target[0]
        true_child = sub_target[1]
        index_true_parent = nodeId2corpusId[true_parent]
        index_true_child = nodeId2corpusId[true_child]
        
        cos_sim_query_parent = cosine_similarities[index_query * len(corpus_embeddings) + index_true_parent]
        cos_sim_query_child = cosine_similarities[index_query * len(corpus_embeddings) + index_true_child]
        
        if cos_sim_query_parent < percentile_80:
            hit_at_1_below_80th.append(isCorrectParentPPRAt1)
            hit_at_5_below_80th.append(isCorrectParentPPRAt5)
            hit_at_10_below_80th.append(isCorrectParentPPRAt10)
            MR_below_80th.append(rank_parent)
        else:
            hit_at_1_above_80th.append(isCorrectParentPPRAt1)
            hit_at_5_above_80th.append(isCorrectParentPPRAt5)
            hit_at_10_above_80th.append(isCorrectParentPPRAt10)
            MR_above_80th.append(rank_parent)
        
        if cos_sim_query_child < percentile_80:
            hit_at_1_below_80th.append(isCorrectChildPPRAt1)
            hit_at_5_below_80th.append(isCorrectChildPPRAt5)
            hit_at_10_below_80th.append(isCorrectChildPPRAt10)
            MR_below_80th.append(rank_child)
        else:
            hit_at_1_above_80th.append(isCorrectChildPPRAt1)
            hit_at_5_above_80th.append(isCorrectChildPPRAt5)
            hit_at_10_above_80th.append(isCorrectChildPPRAt10)
            MR_above_80th.append(rank_child)
        
    
hit_at_1_below_80th = np.mean(hit_at_1_below_80th)
hit_at_5_below_80th = np.mean(hit_at_5_below_80th)
hit_at_10_below_80th = np.mean(hit_at_10_below_80th)
hit_at_1_above_80th = np.mean(hit_at_1_above_80th)
hit_at_5_above_80th = np.mean(hit_at_5_above_80th)
hit_at_10_above_80th = np.mean(hit_at_10_above_80th)
MR_below_80th = np.mean(MR_below_80th)
MR_above_80th = np.mean(MR_above_80th)

print(f"Hit@1 below 80th percentile: {hit_at_1_below_80th}")
print(f"Hit@5 below 80th percentile: {hit_at_5_below_80th}")
print(f"Hit@10 below 80th percentile: {hit_at_10_below_80th}")
print(f"Hit@1 above 80th percentile: {hit_at_1_above_80th}")
print(f"Hit@5 above 80th percentile: {hit_at_5_above_80th}")
print(f"Hit@10 above 80th percentile: {hit_at_10_above_80th}")
print(f"Mean rank below 80th percentile: {MR_below_80th}")
print(f"Mean rank above 80th percentile: {MR_above_80th}")
