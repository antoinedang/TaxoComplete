import networkx as nx
import argparse
import torch
import pickle
import os
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

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
    ) = CPU_Unpickler(f).load()

core_subgraph_undirected = data_prep.core_subgraph.to_undirected()
core_subgraph_undirected.remove_node(data_prep.pseudo_leaf_node)


def get_height(node):
    if len(list(data_prep.core_subgraph.successors(node))) < 2:
        return 0
    else:
        return 1 + max(
            get_height(child) for child in data_prep.core_subgraph.successors(node)
        )


def get_embedding(node_id):
    if node_id == data_prep.pseudo_leaf_node:
        return None
    else:
        return corpus_embeddings[nodeId2corpusId[node_id]]


def get_definition(node_id):
    if node_id == data_prep.pseudo_leaf_node:
        return "N/A"
    else:
        return data_prep.definitions[node_id]


def get_graph_distance(query_node, pred_node, isChild):
    if pred_node != data_prep.pseudo_leaf_node:
        try:
            dist = nx.shortest_path_length(
                data_prep.core_subgraph, source=pred_node, target=query_node
            )
            if isChild:
                dist = -1 * dist
        except nx.NetworkXNoPath:
            try:
                dist = nx.shortest_path_length(
                    core_subgraph_undirected,
                    source=query_node,
                    target=pred_node,
                )
                if not isChild:
                    dist = -1 * dist
            except nx.NetworkXNoPath:
                dist = "N/A"
    else:
        dist = "N/A"
    return dist


def get_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return "N/A"
    else:
        return torch.cosine_similarity(embedding1, embedding2, dim=0)


def is_correct_prediction(edges, target, checkChild, K):
    idx = 1 if checkChild else 0
    return any(
        [
            any(
                [
                    edges[n][idx] == sub_target[idx]
                    or (
                        edges[n][idx] == data_prep.pseudo_leaf_node
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


def get_close_neighborhood(node_id):
    ancestral_nodes = list(
        reversed(
            nx.shortest_path(
                data_prep.core_subgraph, source=data_prep.root, target=node_id
            )
        )
    )
    ancestral_nodes.remove(data_prep.root)
    if node_id in ancestral_nodes:
        ancestral_nodes.remove(node_id)
    children = [
        n
        for n in list(data_prep.core_subgraph.successors(node_id))
        if n != data_prep.pseudo_leaf_node
    ]
    parents = list(data_prep.core_subgraph.predecessors(node_id))
    siblings = set()
    for parent in parents:
        siblings.update(
            [
                n
                for n in data_prep.core_subgraph.successors(parent)
                if n != node_id and n != data_prep.pseudo_leaf_node
            ]
        )
    close_neighborhood = set()
    close_neighborhood.update(ancestral_nodes)
    close_neighborhood.update(children)
    close_neighborhood.update(siblings)
    return close_neighborhood


def get_all_successors(node_id):
    successors = list(data_prep.core_subgraph.successors(node_id))
    for successor in successors:
        successors += get_all_successors(successor)
    return successors


def get_siblings(node_id):
    siblings = set()
    for parent in list(data_prep.core_subgraph.predecessors(node_id)):
        siblings.update(
            [n for n in data_prep.core_subgraph.successors(parent) if n != node_id]
        )
    return siblings


def get_relation_type(source_node, target_node):
    if target_node == data_prep.pseudo_leaf_node:
        return "pseudo_leaf"
    elif target_node in list(data_prep.core_subgraph.successors(source_node)):
        return "child"
    elif target_node in get_all_successors(source_node):
        return "descendant"
    elif source_node in list(data_prep.core_subgraph.successors(target_node)):
        return "parent"
    elif source_node in get_all_successors(target_node):
        return "ancestor"
    elif target_node in get_siblings(source_node):
        return "sibling"
    else:
        return "distant"


csv_format = "queryDef,predChildDef,predParentDef,predChildPPRDef,predParentPPRDef,numCloseNeighbors,queryLevel,queryHeight,isCorrectParentAt1,isCorrectChildAt1,isCorrectParentPPRAt1,isCorrectChildPPRAt1,isCorrectParentAt10,isCorrectChildAt10,isCorrectParentPPRAt10,isCorrectChildPPRAt10,cos_sim_query_pred_child,cos_sim_query_pred_parent,cos_sim_query_pred_child_ppr,cos_sim_query_pred_parent_ppr,graph_dist_query_pred_child,graph_dist_query_pred_parent,graph_dist_query_pred_child_ppr,graph_dist_query_pred_parent_ppr,relation_query_pred_parent,relation_query_pred_child,relation_query_pred_parent_ppr,relation_query_pred_child_ppr,true_parent_def,true_child_def\n"
with open(csv_filename, "w+") as f:
    f.write(csv_format)

#   FOR EACH QUERY:
for i in range(len(data_prep.test_queries)):
    query = data_prep.test_queries[i]
    query_node_id = data_prep.corpusId2nodeId[i]
    query_definition = '"' + str(get_definition(query_node_id)) + '"'
    query_embedding = query_embeddings[i]
    query_level = nx.shortest_path_length(
        data_prep.core_subgraph, source=data_prep.root, target=query_node_id
    )
    query_height = get_height(query_node_id)
    target = targets[i]
    predicted = all_predictions[i]
    predicted_ppr = all_predictions_ppr[i]
    predicted_edge = edges_predictions_test[i][0]
    predicted_edge_ppr = edges_predictions_test_ppr[i][0]
    if predicted_edge[0] == data_prep.pseudo_leaf_node:
        print(" > ERROR: predicted parent is pseudo leaf node")
    if predicted_edge_ppr[0] == data_prep.pseudo_leaf_node:
        print(" > ERROR: PPR predicted parent is pseudo leaf node")
    pred_child_embedding = get_embedding(predicted_edge[1])
    pred_parent_embedding = get_embedding(predicted_edge[0])
    pred_child_embedding_ppr = get_embedding(predicted_edge_ppr[1])
    pred_parent_embedding_ppr = get_embedding(predicted_edge_ppr[0])
    pred_parent_definition = '"' + str(get_definition(predicted_edge[0])) + '"'
    pred_child_definition = '"' + str(get_definition(predicted_edge[1])) + '"'
    pred_parent_ppr_definition = '"' + str(get_definition(predicted_edge_ppr[0])) + '"'
    pred_child_ppr_definition = '"' + str(get_definition(predicted_edge_ppr[1])) + '"'
    random_target_idx = torch.randint(0, len(target), (1,)).item()
    random_target = target[random_target_idx]
    random_gt_parent_definition = '"' + str(get_definition(random_target[0])) + '"'
    random_gt_child_definition = '"' + str(get_definition(random_target[1])) + '"'
    dist_query_pred_parent = get_graph_distance(
        query_node_id, predicted_edge[0], isChild=False
    )
    dist_query_pred_child = get_graph_distance(
        query_node_id, predicted_edge[1], isChild=True
    )
    dist_query_pred_parent_ppr = get_graph_distance(
        query_node_id, predicted_edge_ppr[0], isChild=False
    )
    dist_query_pred_child_ppr = get_graph_distance(
        query_node_id, predicted_edge_ppr[1], isChild=True
    )
    relation_query_pred_parent = get_relation_type(query_node_id, predicted_edge[0])
    relation_query_pred_child = get_relation_type(query_node_id, predicted_edge[1])
    relation_query_pred_parent_ppr = get_relation_type(
        query_node_id, predicted_edge_ppr[0]
    )
    relation_query_pred_child_ppr = get_relation_type(
        query_node_id, predicted_edge_ppr[1]
    )

    # sparsity score: calculate number of nodes in close neighborhood
    close_neighborhood = get_close_neighborhood(query_node_id)
    close_neighborhood_size = len(close_neighborhood)

    # RELEVANCE: isCorrectParent, isCorrectChild, isCorrectParentPPR, isCorrectChildPPR
    isCorrectParentAt1 = is_correct_prediction(
        edges_predictions_test[i], target, checkChild=False, K=1
    )
    isCorrectParentAt10 = is_correct_prediction(
        edges_predictions_test[i], target, checkChild=False, K=10
    )
    isCorrectChildAt1 = is_correct_prediction(
        edges_predictions_test[i], target, checkChild=True, K=1
    )
    isCorrectChildAt10 = is_correct_prediction(
        edges_predictions_test[i], target, checkChild=True, K=10
    )
    isCorrectParentPPRAt1 = is_correct_prediction(
        edges_predictions_test_ppr[i], target, checkChild=False, K=1
    )
    isCorrectParentPPRAt10 = is_correct_prediction(
        edges_predictions_test_ppr[i], target, checkChild=False, K=10
    )
    isCorrectChildPPRAt1 = is_correct_prediction(
        edges_predictions_test_ppr[i], target, checkChild=True, K=1
    )
    isCorrectChildPPRAt10 = is_correct_prediction(
        edges_predictions_test_ppr[i], target, checkChild=True, K=10
    )

    # COSINE SIMILARITY: cos_similarity(query node, predicted parent), cos_similarity(query node, predicted child)
    cos_similarity_query_pred_parent = get_cosine_similarity(
        query_embedding, pred_parent_embedding
    )
    cos_similarity_query_pred_child = get_cosine_similarity(
        query_embedding, pred_child_embedding
    )
    cos_similarity_query_pred_parent_ppr = get_cosine_similarity(
        query_embedding, pred_parent_embedding_ppr
    )
    cos_similarity_query_pred_child_ppr = get_cosine_similarity(
        query_embedding, pred_child_embedding_ppr
    )

    #   STORE IN CSV:
    with open(csv_filename, "a+") as f:
        line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
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
            relation_query_pred_parent,
            relation_query_pred_child,
            relation_query_pred_parent_ppr,
            relation_query_pred_child_ppr,
            random_gt_parent_definition,
            random_gt_child_definition,
        )
        f.write(line)
