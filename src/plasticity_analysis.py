import argparse
import torch
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx
import io
import numpy as np


def _compute_label(
    sampling_method,
    max_graph_distance,
    min_graph_distance,
    node_dist,
    cosine_range=(0, 1),
):

    if sampling_method == "closest":
        label_to_assign = 1 / (node_dist)
    elif sampling_method == "closest_range":
        label_to_assign = 1 / (node_dist)
        mapped_label_to_assign = (
            cosine_range[1] - cosine_range[0]
        ) * label_to_assign  # scale range from length of 1 to the correct length
        mapped_label_to_assign = (
            mapped_label_to_assign + cosine_range[0]
        )  # shift range to start at min
        label_to_assign = mapped_label_to_assign
    elif sampling_method == "closest_range_linear":
        label_to_assign = (max_graph_distance - node_dist) / (
            max_graph_distance - min_graph_distance
        )
        label_to_assign = (
            label_to_assign * (cosine_range[1] - cosine_range[0]) + cosine_range[0]
        )

    return label_to_assign


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


torch.manual_seed(0)
args = argparse.ArgumentParser(description="")
args.add_argument(
    type=str,
    help="Error analysis pickle file path",
    dest="filename",
)
args = args.parse_args()

error_analysis_dir = os.path.dirname(os.path.dirname(args.filename))

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

corpusId2nodeId = {v: k for k, v in nodeId2corpusId.items()}

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


train_node_pairs = [
    (int(x.guid.split("_")[0]), int(x.guid.split("_")[1])) for x in data_prep.trainInput
]

cosine_similarities = []
try:
    for i in range(len(train_node_pairs)):
        print(f"Query {i+1}/{len(train_node_pairs)}")
        node, negn = train_node_pairs[i]
        query_embedding = get_embedding(node)
        corpus_embedding = get_embedding(negn)
        cos_sim = get_cosine_similarity(query_embedding, corpus_embedding)
        if cos_sim != "N/A":
            cosine_similarities.append(float(cos_sim))
        else:
            print("NA")
except Exception as e:
    print(e)

plot_filename = error_analysis_dir + "/mse_distributions.png"
core_subgraph_un = core_subgraph_undirected
max_graph_distance = nx.diameter(core_subgraph_undirected)
min_graph_distance = 1
pseudo_leaf_node = data_prep.pseudo_leaf_node
label_mses = {}
label_configs = [("closest", "")]

for sampling_method in ["closest_range", "closest_range_linear"]:
    for cosine_range in [
        (0, 1),
        (0.161, 0.627),
        (-0.9, 0.9),
        (0.101, 0.161),
        (0, 0.014),
    ]:
        label_configs.append((sampling_method, cosine_range))

for sampling_method, cosine_range in label_configs:
    label_mses[sampling_method + "_" + str(cosine_range)] = []

for i in range(len(cosine_similarities)):
    print(f"Computing data sample #", i, "of", len(cosine_similarities))
    cosine_similarity, nodes = list(zip(cosine_similarities, train_node_pairs))[i]
    node, negn = nodes
    if node == pseudo_leaf_node:
        node_dist = nx.shortest_path_length(core_subgraph_un, negn, negn)
    elif negn == pseudo_leaf_node:
        node_dist = nx.shortest_path_length(core_subgraph_un, node, node)
    else:
        node_dist = nx.shortest_path_length(core_subgraph_un, node, negn)
    for sampling_method, cosine_range in label_configs:
        label = _compute_label(
            sampling_method,
            max_graph_distance,
            min_graph_distance,
            node_dist,
            cosine_range=cosine_range,
        )
        label_mses[sampling_method + "_" + str(cosine_range)].append(
            (label - cosine_similarity) ** 2
        )

for label_name, errors in label_mses.items():
    label_plot_filename = error_analysis_dir + f"/{label_name}_mse_distributions.png"
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    mean_error = np.mean(errors)
    plt.axvline(mean_error, color='red', linestyle='dotted', linewidth=2, label=f"Mean: {mean_error:.5f}")
    plt.xlabel("MSE")
    plt.ylabel("# Node Pairs")
    plt.yscale("log")
    # make x axis limited to 0-1 range
    plt.xlim(0, 1)
    # make y axis limited to 0-10^5 range (log)
    plt.ylim(0, 10**5)
    plt.legend()
    plt.title(f"MSEs on pre-trained model with {label_name} labelling fn")
    plt.savefig(label_plot_filename)
