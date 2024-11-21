import argparse
import torch
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx

torch.manual_seed(0)
args = argparse.ArgumentParser(description="")
args.add_argument(
    type=str,
    help="Error analysis pickle file path",
    dest="filename",
)
args = args.parse_args()

error_analysis_dir = os.path.dirname(args.filename)
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
    ) = pickle.load(f)

core_subgraph_undirected = data_prep.core_subgraph.to_undirected()
core_subgraph_undirected.remove_node(data_prep.pseudo_leaf_node)

# get distances between all pairs of nodes in the graph
distances = []
distances = nx.all_pairs_shortest_path_length(core_subgraph_undirected)

# MAKE A SCATTER PLOT OF THE DISTANCES
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=100)
# change x axis range
plt.xlim(-1, 1)
plt.xlabel("Graph Distance")
plt.ylabel("# Node Pairs")
plt.title("Pairwise distances between nodes in taxonomy")
plt.savefig(plot_filename)


label_plot_filename = error_analysis_dir + "/label_distributions.png"
labeling_fn = lambda d: (1 / d) * (0.5625 + 0.1875) - 0.1875
labels = [labeling_fn(d) for d in distances]

plt.figure(figsize=(10, 6))
plt.hist(labels, bins=100)
# change x axis range
plt.xlim(-1, 1)
plt.xlabel("Label")
plt.ylabel("# Node Pairs")
plt.title("Labels between nodes in taxonomy")
plt.savefig(plot_filename)
