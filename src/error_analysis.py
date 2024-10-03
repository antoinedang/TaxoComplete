import networkx as nx
import argparse
import torch
import numpy as np

import data_process.split_data as st
import data_process.data_loader as dl
from model.sbert import SentenceTransformer
import compute_metrics.metric as ms
from parse_config import ConfigParser
from model.utils import PPRPowerIteration
import os
import pickle

torch.manual_seed(0)
args = argparse.ArgumentParser(description="Training taxonomy expansion model")
args.add_argument(
    "-c",
    "--config",
    default=None,
    type=str,
    help="config file path (default: None)",
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
error_analysis_filename = "error_analysis_variables_{}.pkl".format(name)

if not os.path.exists(error_analysis_filename):
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

    nodeIdsCorpus = [
        data_prep.corpusId2nodeId[idx] for idx in data_prep.corpusId2nodeId
    ]
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

    # SAVE VARIABLES TO PICKLES TO MAKE DEVELOPMENT FASTER
    with open(error_analysis_filename, "wb") as f:
        pickle.dump(
            [
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
            ],
            f,
        )
else:
    with open(error_analysis_filename, "rb") as f:
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
        ) = pickle.load(f)

# prediction_taxonomy = taxonomy.taxonomy
# SET PREDICTION TAXONOMY TO CORE SUBGRAPH BECAUSE THIS IS THE GRAPH THAT TAXOCOMPLETE OPERATES ON
# IF WE USE taxonomy.taxonomy, THE GRAPH DISTANCE BETWEEN QUERY AND PREDICTED PARENT MIGHT BE > 1 EVEN IF THE PREDICTION IS CORRECT
prediction_taxonomy = data_prep.core_subgraph
prediction_taxonomy_undirected = prediction_taxonomy.to_undirected()
prediction_taxonomy_undirected.remove_node(data_prep.pseudo_leaf_node)
taxonomy_roots = (
    [data_prep.root]
    if prediction_taxonomy == data_prep.core_subgraph
    else taxonomy.root
)


def get_height(node):
    if len(list(prediction_taxonomy.successors(node))) < 2:
        return 0
    else:
        return 1 + max(
            get_height(child)
            for child in prediction_taxonomy.successors(node)
            if child != data_prep.pseudo_leaf_node
        )


with open("error_analysis.csv", "w+") as f:
    line = "sizeOfCloseNeighborhood,queryLevel,queryHeight,isCorrectParentAt1,isCorrectChildAt1,isCorrectParentPPRAt1,isCorrectChildPPRAt1,isCorrectParentAt10,isCorrectChildAt10,isCorrectParentPPRAt10,isCorrectChildPPRAt10,cos_sim_query_pred_child,cos_sim_query_pred_parent,cos_sim_query_pred_child_ppr,cos_sim_query_pred_parent_ppr,graph_distance_query_pred_child,graph_distance_query_pred_parent,graph_distance_query_pred_child_ppr,graph_distance_query_pred_parent_ppr\n"
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
    taxonomy_root = None
    for root in taxonomy_roots:
        if nx.has_path(prediction_taxonomy, root, query_node_id):
            taxonomy_root = root
    query_level = nx.shortest_path_length(
        prediction_taxonomy, source=taxonomy_root, target=query_node_id
    )
    query_height = get_height(query_node_id)

    if predicted_edge[0] != data_prep.pseudo_leaf_node:
        try:
            dist_query_pred_parent = nx.shortest_path_length(
                prediction_taxonomy, source=predicted_edge[0], target=query_node_id
            )
        except nx.NetworkXNoPath:
            try:
                dist_query_pred_parent = -1 * nx.shortest_path_length(
                    prediction_taxonomy_undirected,
                    source=query_node_id,
                    target=predicted_edge[0],
                )
            except nx.NetworkXNoPath:
                dist_query_pred_parent = "N/A"
    else:
        dist_query_pred_parent = "N/A"
    if predicted_edge[1] != data_prep.pseudo_leaf_node:
        try:
            dist_query_pred_child = nx.shortest_path_length(
                prediction_taxonomy, source=query_node_id, target=predicted_edge[1]
            )
        except nx.NetworkXNoPath:
            try:
                dist_query_pred_child = -1 * nx.shortest_path_length(
                    prediction_taxonomy_undirected,
                    source=predicted_edge[1],
                    target=query_node_id,
                )
            except nx.NetworkXNoPath:
                dist_query_pred_child = "N/A"
    else:
        dist_query_pred_child = "N/A"
    if predicted_edge_ppr[0] != data_prep.pseudo_leaf_node:
        try:
            dist_query_pred_parent_ppr = nx.shortest_path_length(
                prediction_taxonomy,
                source=predicted_edge_ppr[0],
                target=query_node_id,
            )
        except nx.NetworkXNoPath:
            try:
                dist_query_pred_parent_ppr = -1 * nx.shortest_path_length(
                    prediction_taxonomy_undirected,
                    source=query_node_id,
                    target=predicted_edge_ppr[0],
                )
            except nx.NetworkXNoPath:
                dist_query_pred_parent_ppr = "N/A"
    else:
        dist_query_pred_parent_ppr = "N/A"
    if predicted_edge_ppr[1] != data_prep.pseudo_leaf_node:
        try:
            dist_query_pred_child_ppr = nx.shortest_path_length(
                prediction_taxonomy, source=query_node_id, target=predicted_edge_ppr[1]
            )
        except nx.NetworkXNoPath:
            try:
                dist_query_pred_child_ppr = -1 * nx.shortest_path_length(
                    prediction_taxonomy_undirected,
                    source=predicted_edge_ppr[1],
                    target=query_node_id,
                )
            except nx.NetworkXNoPath:
                dist_query_pred_child_ppr = "N/A"
    else:
        dist_query_pred_child_ppr = "N/A"

    # sparsity score: calculate number of nodes in close neighborhood
    ancestral_nodes = list(
        reversed(
            nx.shortest_path(
                prediction_taxonomy, source=taxonomy_root, target=query_node_id
            )
        )
    )
    ancestral_nodes.remove(taxonomy_root)
    if query_node_id in ancestral_nodes:
        ancestral_nodes.remove(query_node_id)
    children = [
        n
        for n in list(prediction_taxonomy.successors(query_node_id))
        if n != data_prep.pseudo_leaf_node
    ]
    parents = list(prediction_taxonomy.predecessors(query_node_id))
    siblings = set()
    for parent in parents:
        siblings.update(
            [
                n
                for n in prediction_taxonomy.successors(parent)
                if n != query_node_id and n != data_prep.pseudo_leaf_node
            ]
        )
    close_neighborhood = set()
    close_neighborhood.update(ancestral_nodes)
    close_neighborhood.update(children)
    close_neighborhood.update(siblings)
    close_neighborhood_size = len(close_neighborhood)

    # RELEVANCE: isCorrectParent, isCorrectChild, isCorrectParentPPR, isCorrectChildPPR
    isCorrectParentAt1 = any(
        [predicted_edge[0] == sub_target[0] for sub_target in target]
    )
    isCorrectChildAt1 = any(
        [
            predicted_edge[1] == sub_target[1]
            or (
                predicted_edge[1] == data_prep.pseudo_leaf_node
                and not (sub_target[1] in list(prediction_taxonomy.nodes()))
            )
            for sub_target in target
        ]
    )
    isCorrectParentPPRAt1 = any(
        [predicted_edge_ppr[0] == sub_target[0] for sub_target in target]
    )
    isCorrectChildPPRAt1 = any(
        [
            predicted_edge_ppr[1] == sub_target[1]
            or (
                predicted_edge_ppr[1] == data_prep.pseudo_leaf_node
                and not (sub_target[1] in list(prediction_taxonomy.nodes()))
            )
            for sub_target in target
        ]
    )

    isCorrectParentAt10 = any(
        [
            any(
                [
                    edges_predictions_test[i][n][0] == sub_target[0]
                    for sub_target in target
                ]
            )
            for n in range(10)
        ]
    )
    isCorrectChildAt10 = any(
        [
            any(
                [
                    edges_predictions_test[i][n][1] == sub_target[1]
                    or (
                        edges_predictions_test[i][n][1] == data_prep.pseudo_leaf_node
                        and not (sub_target[1] in list(prediction_taxonomy.nodes()))
                    )
                    for sub_target in target
                ]
            )
            for n in range(10)
        ]
    )
    isCorrectParentPPRAt10 = any(
        [
            any(
                [
                    edges_predictions_test_ppr[i][n][0] == sub_target[0]
                    for sub_target in target
                ]
            )
            for n in range(10)
        ]
    )
    isCorrectChildPPRAt10 = any(
        [
            any(
                [
                    edges_predictions_test_ppr[i][n][1] == sub_target[1]
                    or (
                        edges_predictions_test_ppr[i][n][1]
                        == data_prep.pseudo_leaf_node
                        and not (sub_target[1] in list(prediction_taxonomy.nodes()))
                    )
                    for sub_target in target
                ]
            )
            for n in range(10)
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
        line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            close_neighborhood_size,
            query_level,
            query_height,
            isCorrectParentAt1,
            isCorrectChildAt1,
            isCorrectParentPPRAt1,
            isCorrectChildPPRAt1,
            isCorrectParentAt10,
            isCorrectChildAt10,
            isCorrectParentPPRAt10,
            isCorrectChildPPRAt10,
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
