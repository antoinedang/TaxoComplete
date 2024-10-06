import networkx as nx
import argparse
import torch
import pickle
import os

torch.manual_seed(0)
args = argparse.ArgumentParser(description="Run error analysis on evaluation results.")
args.add_argument(
    type=str,
    help="Error analysis pickle file path",
    dest="filename",
)
args = args.parse_args()

error_analysis_dir = os.path.dirname(args.filename)
csv_filename = error_analysis_dir + "/error_analysis.csv"

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


csv_format = "queryDef,predChildDef,predParentDef,predChildPPRDef,predParentPPRDef,numCloseNeighbors,queryLevel,queryHeight,isCorrectParentAt1,isCorrectChildAt1,isCorrectParentPPRAt1,isCorrectChildPPRAt1,isCorrectParentAt10,isCorrectChildAt10,isCorrectParentPPRAt10,isCorrectChildPPRAt10,cos_sim_query_pred_child,cos_sim_query_pred_parent,cos_sim_query_pred_child_ppr,cos_sim_query_pred_parent_ppr,graph_dist_query_pred_child,graph_dist_query_pred_parent,graph_dist_query_pred_child_ppr,graph_dist_query_pred_parent_ppr\n"
with open(csv_filename, "w+") as f:
    f.write(csv_format)

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

    query_definition = data_prep.corpusId2definition[nodeId2corpusId[query_node_id]]

    if predicted_edge[0] != data_prep.pseudo_leaf_node:
        pred_parent_definition = data_prep.corpusId2definition[
            nodeId2corpusId[predicted_edge[0]]
        ]
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
        print(" > ERROR: predicted parent is pseudo leaf node")
        pred_parent_definition = "N/A"
        dist_query_pred_parent = "N/A"
    if predicted_edge[1] != data_prep.pseudo_leaf_node:
        pred_child_definition = data_prep.corpusId2definition[
            nodeId2corpusId[predicted_edge[1]]
        ]
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
        pred_child_definition = "N/A"
    if predicted_edge_ppr[0] != data_prep.pseudo_leaf_node:
        pred_parent_ppr_definition = data_prep.corpusId2definition[
            nodeId2corpusId[predicted_edge_ppr[0]]
        ]
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
        print(" > ERROR: PPR predicted parent is pseudo leaf node")
        pred_parent_ppr_definition = "N/A"
        dist_query_pred_parent_ppr = "N/A"
    if predicted_edge_ppr[1] != data_prep.pseudo_leaf_node:
        pred_child_ppr_definition = data_prep.corpusId2definition[
            nodeId2corpusId[predicted_edge_ppr[1]]
        ]
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
        pred_child_ppr_definition = "N/A"
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
    with open(csv_filename, "a+") as f:
        line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            query_definition,
            pred_child_definition,
            pred_parent_definition,
            pred_child_ppr_definition,
            pred_parent_ppr_definition,
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
