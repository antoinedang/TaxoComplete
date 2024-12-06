import networkx as nx
import math
import argparse
import torch
from torch.utils.data import DataLoader
import data_process.split_data as st
import data_process.data_loader as dl
import data_process.helpers as hp
from model.sbert import SentenceTransformer, losses
from model.sbert.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
import compute_metrics.metric as ms
from parse_config import ConfigParser
from model.utils import PPRPowerIteration
from model.sbert.losses import exp_map_hyperboloid, hyperbolic_cosine_similarity
from sentence_transformers import util
import os
import pickle
import geoopt
import transformers

torch.manual_seed(0)
args = argparse.ArgumentParser(description="Training taxonomy expansion model")
args.add_argument(
    "-c", "--config", default=None, type=str, help="config file path (default: None)"
)
args.add_argument(
    "-n",
    "--name",
    default=None,
    type=str,
    help="Folder to save .pkl to (default: None)",
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
batch_size = config["batch_size"]
epochs = config["epochs"]
alpha = config["alpha"]

cosine_range = config.get("cossim_mapping_range", [0, 1])
loss_alpha = float(config.get("loss_alpha", 1))
loss_beta = float(config.get("loss_beta", 0))

device = "cuda" if torch.cuda.is_available() else "cpu"
target_device = torch.device(device)

taxonomy = dl.TaxoDataset(
    name, data_path, raw=True, partition_pattern=partition_pattern, seed=seed
)
data_prep = st.Dataset(
    taxonomy,
    sampling_method,
    neg_number,
    seed,
    cosine_range=cosine_range,
)

model_name = config["model_name"]
if torch.cuda.is_available():
    model = SentenceTransformer.SentenceTransformer(model_name, device="cuda")
else:
    model = SentenceTransformer.SentenceTransformer(model_name)

if config.get("cossim_mapping_range_percentile", None) is not None:
    corpus_embeddings = model.encode(
        data_prep.corpus, convert_to_tensor=True, show_progress_bar=True
    )
    query_embeddings = model.encode(data_prep.test_queries, convert_to_tensor=True)
    cosine_range = hp.compute_cosine_ranges(
        config.get("cossim_mapping_range_percentile"),
        query_embeddings,
        corpus_embeddings,
    )
    data_prep = st.Dataset(
        taxonomy,
        sampling_method,
        neg_number,
        seed,
        cosine_range=cosine_range,
    )

g = torch.Generator()
g.manual_seed(0)


nodeIdsCorpus = [data_prep.corpusId2nodeId[idx] for idx in data_prep.corpusId2nodeId]
core_graph = data_prep.core_subgraph.copy()
core_graph.remove_node(data_prep.pseudo_leaf_node)
nodes_core_subgraph = list(core_graph.nodes)
assert nodes_core_subgraph == nodeIdsCorpus
propagation = PPRPowerIteration(
    nx.adjacency_matrix(core_graph), alpha=alpha, niter=10
).to(target_device)


# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(data_prep.trainInput, shuffle=True, batch_size=batch_size)
warmup_steps = math.ceil(
    len(train_dataloader) * epochs * 0.1
)  # 10% of train data for warm-up
train_loss = losses.CosineSimilarityLoss(
    model,
    alpha=loss_alpha,
    beta=loss_beta,
    modified_loss=bool(config.get("loss_modified", "false") == "true"),
    hyperbolic=bool(config.get("hyperbolic", "false") == "true"),
    hyperbolic_curvature=float(config.get("hyperbolic_curvature", 1.0)),
    cosine_absolute=bool(config.get("cosine_absolute", "false") == "true"),
)
if config.get("hyperbolic", "false") == "true":
    optimizer_class = geoopt.optim.RiemannianAdam

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        data_prep.val_examples,
        name="sts-dev",
        main_similarity=SimilarityFunction.HYPERBOLIC,
        hyperbolic_c=float(config.get("hyperbolic_curvature", 1.0)),
    )
else:
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        data_prep.val_examples, name="sts-dev"
    )
    optimizer_class = transformers.AdamW
# Tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    evaluation_steps=1000,
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path=str(config.save_dir),
    save_best_model=True,
    optimizer_class=optimizer_class,
)

model = SentenceTransformer.SentenceTransformer(str(config.save_dir))
corpus_embeddings = model.encode(
    data_prep.corpus, convert_to_tensor=True, show_progress_bar=True
)
preds = propagation(
    corpus_embeddings, torch.tensor(range(len(nodeIdsCorpus)), device=target_device)
)

if config.get("hyperbolic", "false") == "true":
    c = config.get("hyperbolic_curvature", 1.0)
    corpus_embeddings = exp_map_hyperboloid(corpus_embeddings, c)
    score_function = lambda x, y: hyperbolic_cosine_similarity(x, y, c)
else:
    score_function = util.cos_sim

if config.get("cosine_absolute", "false") == "true":
    score_function = lambda x, y: torch.abs(score_function(x, y))

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
    score_function,
)
ms.save_results(
    str(config.save_dir) + "/", all_targets_val, edges_predictions_val, "eval_val"
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
    score_function,
)
ms.save_results(
    str(config.save_dir) + "/", all_targets_test, edges_predictions_test, "eval_test"
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
    score_function,
)
ms.save_results(
    str(config.save_dir) + "/",
    all_targets_val_ppr,
    edges_predictions_val_ppr,
    "eval_val_ppr",
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
    score_function,
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
    + f"/{args.name if args.name is not None else ''}"
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
