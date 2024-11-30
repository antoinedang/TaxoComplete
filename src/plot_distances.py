import argparse
import torch
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx
import io
import numpy as np


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
plot_filename = error_analysis_dir + "/graph_distance_distributions.png"

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

# get distances between all pairs of nodes in the graph
print("Computing graph distances...")
all_distances = dict(nx.all_pairs_shortest_path_length(core_subgraph_undirected))
distances = []
test_query_node_ids = [
    data_prep.corpusId2nodeId[i] for i in range(len(data_prep.test_queries))
]
for i in range(len(data_prep.test_queries)):
    print(f"Query {i+1}/{len(data_prep.test_queries)}")
    for n in core_subgraph_undirected.nodes:
        if n != data_prep.corpusId2nodeId[i]:
            distances.append(all_distances[data_prep.corpusId2nodeId[i]][n])

# MAKE A SCATTER PLOT OF THE DISTANCES
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=50)
# change x axis range
plt.xlabel("Graph Distance")
plt.ylabel("# Node Pairs")
plt.title("Pairwise distances between nodes in taxonomy")
plt.savefig(plot_filename)


def scaled_label(x, lower, upper):
    return (upper - lower) * (1 / x) + lower


max_graph_distance = nx.diameter(core_subgraph_undirected)
min_graph_distance = 1


def linear_scaled_label(x, lower, upper):
    label_to_assign = (max_graph_distance - x) / (
        max_graph_distance - min_graph_distance
    )
    label_to_assign = label_to_assign * (upper - lower) + lower
    return label_to_assign


for label_name, labeling_fn in [
    ("original", lambda d: 1 / d),
    ("epsilon", lambda d: 1 / (d + 0.1)),
    ("linear", lambda d: max(-0.9, 1 - (0.1 * d))),
    ("boundary", lambda d: scaled_label(d, 0, 0.9)),
    ("full_boundary", lambda d: scaled_label(d, -0.9, 0.9)),
    ("min_max", lambda d: scaled_label(d, -0.1875, 0.5625)),
    ("min_max_mistake", lambda d: scaled_label(d, 0.1875, 0.9375)),
    ("linear_scale", lambda d: linear_scaled_label(d, -0.1875, 0.5625)),
]:
    label_plot_filename = error_analysis_dir + f"/{label_name}_label_distributions.png"
    print(f"Computing {label_name} labels...")
    labels = [labeling_fn(d) for d in distances]
    # labels.append(labeling_fn(min(distances)-0.9))
    # labels.append(labeling_fn(max(distances)+1))
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=50)
    # change x axis range
    plt.xlim(-1, 1)
    plt.xlabel("Label")
    plt.ylabel("# Node Pairs")
    # make y axis log scale
    plt.yscale("log")
    plt.title(f"Labels between nodes in taxonomy ({label_name} label fn)")
    plt.savefig(label_plot_filename)
