import argparse
import torch
import pickle
import os
import matplotlib.pyplot as plt

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
    ) = pickle.load(f)

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


# MAKE A SCATTER PLOT OF THE COSINE SIMILARITIES
plt.figure(figsize=(10, 6))
plt.hist(cosine_similarities, bins=100)
# change x axis range
plt.xlim(-1, 1)
plt.xlabel("Cosine similarity")
plt.ylabel("# Query/Node Pairs")
plt.title("Cosine similarities between query and corpus embeddings")
plt.savefig(plot_filename)
