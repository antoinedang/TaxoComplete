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
import pickle
import os

torch.manual_seed(0)
args = argparse.ArgumentParser(description="Training taxonomy expansion model")
args.add_argument(
    "-c", "--config", default=None, type=str, help="config file path (default: None)"
)
args.add_argument(
    "--untrained",
    action="store_true",
    help="whether to use the untrained model to evaluate",
)
args.add_argument(
    "--random",
    action="store_true",
    help="whether to use a purely random model to evaluate",
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

if args.untrained:
    model = SentenceTransformer.SentenceTransformer(model_name)
else:
    model = SentenceTransformer.SentenceTransformer(config["model_path"])
if args.random:
    corpus_embeddings = torch.rand(
        len(data_prep.corpus), model.get_sentence_embedding_dimension()
    ).to(target_device)
else:
    corpus_embeddings = model.encode(
        data_prep.corpus, convert_to_tensor=True, show_progress_bar=True
    )
preds = propagation(
    corpus_embeddings, torch.tensor(range(len(nodeIdsCorpus)), device=target_device)
)

(
    all_targets_val,
    all_predictions_val,
    all_scores_val,
    edges_predictions_val,
    all_edges_scores_val,
) = ms.compute_prediction(
    data_prep.core_subgraph.edges,
    data_prep.pseudo_leaf_node,
    data_prep.valid_queries,
    corpus_embeddings,
    model,
    data_prep.valid_node_list,
    data_prep.valid_node2pos,
    data_prep.corpusId2nodeId,
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
    all_targets_val_ppr,
    all_predictions_val_ppr,
    all_scores_val_ppr,
    edges_predictions_val_ppr,
    all_edges_scores_val_ppr,
) = ms.compute_prediction(
    data_prep.core_subgraph.edges,
    data_prep.pseudo_leaf_node,
    data_prep.valid_queries,
    preds,
    model,
    data_prep.valid_node_list,
    data_prep.valid_node2pos,
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

ms.save_results(
    str(config.save_dir) + "/", all_targets_val, edges_predictions_val, "eval_val"
)
ms.save_results(
    str(config.save_dir) + "/", all_targets_test, edges_predictions_test, "eval_test"
)
ms.save_results(
    str(config.save_dir) + "/",
    all_targets_val_ppr,
    edges_predictions_val_ppr,
    "eval_val_ppr",
)
ms.save_results(
    str(config.save_dir) + "/",
    all_targets_test_ppr,
    edges_predictions_test_ppr,
    "eval_test_ppr",
)


# SAVE VARIABLES TO PICKLES TO MAKE DEVELOPMENT FASTER
targets = [data_prep.test_node2pos[node] for node in data_prep.test_node_list]
query_embeddings = model.encode(data_prep.test_queries, convert_to_tensor=True)
nodeId2corpusId = {v: k for k, v in data_prep.corpusId2nodeId.items()}

pickle_folder = (
    os.path.dirname(str(config.save_dir))
    + f"/{'random' if args.random else ('untrained' if args.untrained else 'trained')}"
)

os.makedirs(
    pickle_folder,
    exist_ok=True,
)
error_analysis_filename = pickle_folder + "/error_analysis.pkl"

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
