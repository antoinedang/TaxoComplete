import networkx as nx
import math
import argparse
import torch
import numpy as np
import time

from torch.utils.data import DataLoader
import data_process.split_data as st
import data_process.data_loader as dl
from model.sbert import SentenceTransformer, losses
from model.sbert.evaluation import EmbeddingSimilarityEvaluator
import compute_metrics.metric as ms
from parse_config import ConfigParser
from model.utils import PPRPowerIteration

torch.manual_seed(0)
args = argparse.ArgumentParser(description="Training taxonomy expansion model")
args.add_argument(
    "-c", "--config", default=None, type=str, help="config file path (default: None)"
)
config = ConfigParser(args)
args = args.parse_args()

saving_path = config["saving_path"]
name = config["name"]
data_path = config["data_path"]
sampling_method = config["sampling"]
neg_number = config["neg_number"]
partition_pattern = config["partition_pattern"]
seed = config["seed"]


taxonomy = dl.TaxoDataset(
    name, data_path, raw=True, partition_pattern=partition_pattern, seed=seed
)
data_prep = st.Dataset(taxonomy, sampling_method, neg_number, seed)
model_name = config["model_name"]

device = "cuda" if torch.cuda.is_available() else "cpu"
target_device = torch.device(device)

g = torch.Generator()
g.manual_seed(0)


batch_size = config["batch_size"]
epochs = config["epochs"]

alpha = config["alpha"]

nclasses = max(list(data_prep.trainInputLevel.values())) + 1
nodes_list = np.array(data_prep.core_subgraph.nodes())

nodeIdsCorpus = [data_prep.corpusId2nodeId[idx] for idx in data_prep.corpusId2nodeId]
core_graph = data_prep.core_subgraph.copy()
core_graph.remove_node(data_prep.pseudo_leaf_node)
nodes_core_subgraph = list(core_graph.nodes)
assert nodes_core_subgraph == nodeIdsCorpus
propagation = PPRPowerIteration(
    nx.adjacency_matrix(core_graph), alpha=alpha, niter=10
).to(target_device)


model = SentenceTransformer.SentenceTransformer(config["model_path"])
corpus_embeddings = model.encode(
    data_prep.corpus, convert_to_tensor=True, show_progress_bar=True
)
preds = propagation(
    corpus_embeddings, torch.tensor(range(len(nodeIdsCorpus)), device=target_device)
)

(
    all_targets_test,
    all_predictions,
    all_scores_test,
    edges_predictions_test,
    all_edges_scores_test,
) = ms.compute_prediction(
    data_prep.core_subgraph.edges,
    data_prep.pseudo_leaf_node,
    data_prep.test_queries,
    corpus_embeddings,
    model,
    data_prep.test_node_list,
    data_prep.test_node2pos,
    data_prep.corpusId2nodeId,
)

(
    all_targets_test_ppr,
    all_predictions_ppr,
    all_scores_test_ppr,
    edges_predictions_test_ppr,
    all_edges_scores_test_ppr,
) = ms.compute_prediction(
    data_prep.core_subgraph.edges,
    data_prep.pseudo_leaf_node,
    data_prep.test_queries,
    preds,
    model,
    data_prep.test_node_list,
    data_prep.test_node2pos,
    data_prep.corpusId2nodeId,
)

targets = [data_prep.test_node2pos[node] for node in data_prep.test_node_list]
query_embeddings = model.encode(data_prep.test_queries, convert_to_tensor=True)
nodeId2corpusId = {v: k for k, v in data_prep.corpusId2nodeId.items()}
undirected_core_subgraph = data_prep.core_subgraph.to_undirected()
undirected_core_subgraph.remove_node(data_prep.pseudo_leaf_node)


def get_height(node):
    if not list(core_graph.successors(node)):
        return 0
    else:
        return 1 + max(get_height(child) for child in core_graph.successors(node))


with open("error_analysis.csv", "w+") as f:
    line = "sizeOfCloseNeighborhood,queryLevel,queryHeight,isCorrectParent,isCorrectChild,isCorrectParentPPR,isCorrectChildPPR,cos_sim_query_pred_child,cos_sim_query_pred_parent,cos_sim_query_pred_child_ppr,cos_sim_query_pred_parent_ppr,graph_distance_query_pred_child,graph_distance_query_pred_parent,graph_distance_query_pred_child_ppr,graph_distance_query_pred_parent_ppr\n"
    f.write(line)

#   FOR EACH QUERY:
for i in range(len(data_prep.test_queries)):
    print(" >>>>>> Query", i + 1, "out of", len(data_prep.test_queries))
    query = data_prep.test_queries[i]
    query_node_id = data_prep.corpusId2nodeId[i]
    query_embedding = query_embeddings[i]
    target = targets[i]
    predicted = all_predictions[i]
    predicted_ppr = all_predictions_ppr[i]
    predicted_edge = edges_predictions_test[i][0]
    if predicted_edge[1] == data_prep.pseudo_leaf_node:
        pred_child_embedding = None
    else:
        pred_child_embedding = corpus_embeddings[nodeId2corpusId[predicted_edge[1]]]
    if predicted_edge[0] == data_prep.pseudo_leaf_node:
        pred_parent_embedding = None
    else:
        pred_parent_embedding = corpus_embeddings[nodeId2corpusId[predicted_edge[0]]]
    predicted_edge_ppr = edges_predictions_test_ppr[i][0]
    if predicted_edge_ppr[1] == data_prep.pseudo_leaf_node:
        pred_child_embedding_ppr = None
    else:
        pred_child_embedding_ppr = preds[nodeId2corpusId[predicted_edge_ppr[1]]]
    if predicted_edge_ppr[0] == data_prep.pseudo_leaf_node:
        pred_parent_embedding_ppr = None
    else:
        pred_parent_embedding_ppr = preds[nodeId2corpusId[predicted_edge_ppr[0]]]

    # NX: sparsityScore, level, height, graph_distance(query node, predicted parent), graph_distance(query node, predicted child)
    query_level = nx.shortest_path_length(
        undirected_core_subgraph, source=data_prep.root, target=query_node_id
    )
    query_height = get_height(query_node_id)

    if predicted_edge[0] != data_prep.pseudo_leaf_node:
        dist_query_pred_parent = nx.shortest_path_length(
            undirected_core_subgraph, source=query_node_id, target=predicted_edge[0]
        )
    else:
        dist_query_pred_parent = "N/A"
    if predicted_edge[1] != data_prep.pseudo_leaf_node:
        dist_query_pred_child = nx.shortest_path_length(
            undirected_core_subgraph, source=query_node_id, target=predicted_edge[1]
        )
    else:
        dist_query_pred_child = "N/A"
    if predicted_edge_ppr[0] != data_prep.pseudo_leaf_node:
        dist_query_pred_parent_ppr = nx.shortest_path_length(
            undirected_core_subgraph, source=query_node_id, target=predicted_edge_ppr[0]
        )
    else:
        dist_query_pred_parent_ppr = "N/A"
    if predicted_edge_ppr[1] != data_prep.pseudo_leaf_node:
        dist_query_pred_child_ppr = nx.shortest_path_length(
            undirected_core_subgraph, source=query_node_id, target=predicted_edge_ppr[1]
        )
    else:
        dist_query_pred_child_ppr = "N/A"
    # sparsity score: calculate number of nodes in close neighborhood
    ancestral_nodes = list(
        reversed(
            nx.shortest_path(
                undirected_core_subgraph, source=data_prep.root, target=query_node_id
            )
        )
    )
    ancestral_nodes.remove(data_prep.root)
    ancestral_nodes.remove(query_node_id)
    children = [
        n
        for n in list(data_prep.core_subgraph.successors(query_node_id))
        if n != data_prep.pseudo_leaf_node
    ]
    parent = list(data_prep.core_subgraph.predecessors(query_node_id))[0]
    siblings = [
        n
        for n in data_prep.core_subgraph.successors(parent)
        if n != query_node_id and n != data_prep.pseudo_leaf_node
    ]
    close_neighborhood_size = len(ancestral_nodes) + len(children) + len(siblings)

    # RELEVANCE: isCorrectParent, isCorrectChild, isCorrectParentPPR, isCorrectChildPPR
    isCorrectParent = any([predicted_edge[0] == sub_target[0] for sub_target in target])
    isCorrectChild = any(
        [
            predicted_edge[1] == sub_target[1]
            or (
                predicted_edge[1] == data_prep.pseudo_leaf_node
                and not (sub_target[1] in list(undirected_core_subgraph.nodes()))
            )
            for sub_target in target
        ]
    )
    isCorrectParentPPR = any(
        [predicted_edge_ppr[0] == sub_target[0] for sub_target in target]
    )
    isCorrectChildPPR = any(
        [
            predicted_edge_ppr[1] == sub_target[1]
            or (
                predicted_edge_ppr[1] == data_prep.pseudo_leaf_node
                and not (sub_target[1] in list(undirected_core_subgraph.nodes()))
            )
            for sub_target in target
        ]
    )

    # COSINE SIMILARITY: cos_similarity(query node, predicted parent), cos_similarity(query node, predicted child)
    if pred_parent_embedding is None:
        cos_similarity_query_pred_parent = "N/A"
    else:
        cos_similarity_query_pred_parent = torch.cosine_similarity(
            query_embedding, pred_parent_embedding, dim=0
        )
    if pred_child_embedding is None:
        cos_similarity_query_pred_child = "N/A"
    else:
        cos_similarity_query_pred_child = torch.cosine_similarity(
            query_embedding, pred_child_embedding, dim=0
        )
    if pred_parent_embedding_ppr is None:
        cos_similarity_query_pred_parent_ppr = "N/A"
    else:
        cos_similarity_query_pred_parent_ppr = torch.cosine_similarity(
            query_embedding, pred_parent_embedding_ppr, dim=0
        )
    if pred_child_embedding_ppr is None:
        cos_similarity_query_pred_child_ppr = "N/A"
    else:
        cos_similarity_query_pred_child_ppr = torch.cosine_similarity(
            query_embedding, pred_child_embedding_ppr, dim=0
        )

    #   STORE IN CSV:
    with open("error_analysis.csv", "a+") as f:
        line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            close_neighborhood_size,
            query_level,
            query_height,
            isCorrectParent,
            isCorrectChild,
            isCorrectParentPPR,
            isCorrectChildPPR,
            cos_similarity_query_pred_child,
            cos_similarity_query_pred_parent,
            cos_similarity_query_pred_child_ppr,
            cos_similarity_query_pred_parent_ppr,
            dist_query_pred_child,
            dist_query_pred_parent,
            dist_query_pred_child_ppr,
            dist_query_pred_parent_ppr,
        )
        f.write(line)
