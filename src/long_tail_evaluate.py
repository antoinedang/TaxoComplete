import argparse
import torch
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


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
    ) = CPU_Unpickler(f).load()

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


if not os.path.exists(error_analysis_dir + "/cosine_similarities.pkl"):
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

    with open(error_analysis_dir + "/cosine_similarities.pkl", "wb") as f:
        pickle.dump(cosine_similarities, f)
else:
    with open(error_analysis_dir + "/cosine_similarities.pkl", "rb") as f:
        cosine_similarities = pickle.load(f)


def is_correct_prediction(edges, target, checkChild, K):
    idx = 1 if checkChild else 0
    return any(
        [
            any(
                [
                    edges[i][n][idx] == sub_target[idx]
                    or (
                        edges[i][n][idx] == data_prep.pseudo_leaf_node
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


def rank_of_correct_prediction(edges, target, checkChild):
    rank = 0
    found_correct = False
    while not found_correct:
        rank += 1
        found_correct = is_correct_prediction(edges, target, checkChild, rank)
    return rank


cosine_similarities_sorted = np.sort(cosine_similarities)
if not os.path.exists(error_analysis_dir + "/evaluations_per_cosine_sim.pkl"):
    hit_at_1 = []
    hit_at_5 = []
    hit_at_10 = []
    MR = []
    cosine_similarity_score = []
    #   FOR EACH QUERY:
    for i in range(len(data_prep.test_queries)):
        print("Evaluating query #{} of {}".format(i, len(data_prep.test_queries)))
        query = data_prep.test_queries[i]
        target = targets[i]

        # RELEVANCE: isCorrectParent, isCorrectChild, isCorrectParentPPR, isCorrectChildPPR
        isCorrectParentPPRAt1 = is_correct_prediction(
            edges_predictions_test_ppr, target, checkChild=False, K=1
        )
        isCorrectParentPPRAt5 = is_correct_prediction(
            edges_predictions_test_ppr, target, checkChild=False, K=5
        )
        isCorrectParentPPRAt10 = is_correct_prediction(
            edges_predictions_test_ppr, target, checkChild=False, K=10
        )
        isCorrectChildPPRAt1 = is_correct_prediction(
            edges_predictions_test_ppr, target, checkChild=True, K=1
        )
        isCorrectChildPPRAt5 = is_correct_prediction(
            edges_predictions_test_ppr, target, checkChild=True, K=5
        )
        isCorrectChildPPRAt10 = is_correct_prediction(
            edges_predictions_test_ppr, target, checkChild=True, K=10
        )
        rank_child = rank_of_correct_prediction(
            edges_predictions_test_ppr, target, checkChild=True
        )
        rank_parent = rank_of_correct_prediction(
            edges_predictions_test_ppr, target, checkChild=False
        )
        for sub_target in target:
            # COSINE SIMILARITY: cos_similarity_untrained(query node, actual parent), cos_similarity_untrained(query node, actual child)
            index_query = data_prep.test_queries.index(query)
            true_parent = sub_target[0]
            index_true_parent = nodeId2corpusId[true_parent]
            cos_sim_query_parent = cosine_similarities[
                index_query * len(corpus_embeddings) + index_true_parent
            ]
            hit_at_1.append(isCorrectParentPPRAt1)
            hit_at_5.append(isCorrectParentPPRAt5)
            hit_at_10.append(isCorrectParentPPRAt10)
            MR.append(rank_parent)
            cosine_similarity_score.append(cos_sim_query_parent)

            true_child = sub_target[1]
            if true_child == data_prep.pseudo_leaf_node:
                continue
            index_true_child = nodeId2corpusId[true_child]
            cos_sim_query_child = cosine_similarities[
                index_query * len(corpus_embeddings) + index_true_child
            ]
            hit_at_1.append(isCorrectChildPPRAt1)
            hit_at_5.append(isCorrectChildPPRAt5)
            hit_at_10.append(isCorrectChildPPRAt10)
            MR.append(rank_child)
            cosine_similarity_score.append(cos_sim_query_child)

    with open(error_analysis_dir + "/evaluations_per_cosine_sim.pkl", "wb") as f:
        pickle.dump((hit_at_1, hit_at_5, hit_at_10, MR, cosine_similarity_score), f)
else:
    with open(error_analysis_dir + "/evaluations_per_cosine_sim.pkl", "rb") as f:
        hit_at_1, hit_at_5, hit_at_10, MR, cosine_similarity_score = pickle.load(f)

labels = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]
hit1 = []
hit5 = []
hit10 = []
mean_rank = []

for i in range(10):
    if i == 0:
        percentile_lower_bound = np.min(cosine_similarity_score)
    else:
        percentile_lower_bound = np.percentile(cosine_similarity_score, (i * 10))
    if i == 9:
        percentile_upper_bound = np.max(cosine_similarity_score)
    else:
        percentile_upper_bound = np.percentile(cosine_similarity_score, (i + 1) * 10)

    hit_at_1_average = np.mean(
        [
            hit_at_1[j]
            for j in range(len(hit_at_1))
            if percentile_lower_bound
            <= cosine_similarity_score[j]
            <= percentile_upper_bound
        ]
    )
    hit_at_5_average = np.mean(
        [
            hit_at_5[j]
            for j in range(len(hit_at_5))
            if percentile_lower_bound
            <= cosine_similarity_score[j]
            <= percentile_upper_bound
        ]
    )
    hit_at_10_average = np.mean(
        [
            hit_at_10[j]
            for j in range(len(hit_at_10))
            if percentile_lower_bound
            <= cosine_similarity_score[j]
            <= percentile_upper_bound
        ]
    )
    MR_average = np.mean(
        [
            MR[j]
            for j in range(len(MR))
            if percentile_lower_bound
            <= cosine_similarity_score[j]
            <= percentile_upper_bound
        ]
    )

    # print(f"Percentile range: {percentile_lower_bound} - {percentile_upper_bound} ({i*10}-{(i+1)*10}%)")
    # print(f"Hit@1: {hit_at_1_average}")
    # print(f"Hit@5: {hit_at_5_average}")
    # print(f"Hit@10: {hit_at_10_average}")
    # print(f"Mean rank: {MR_average}")

    hit1.append(hit_at_1_average)
    hit5.append(hit_at_5_average)
    hit10.append(hit_at_10_average)
    mean_rank.append(MR_average)


plt.figure(figsize=(10, 6))
plt.plot(labels, hit1, label="Hit@1")
plt.plot(labels, hit5, label="Hit@5")
plt.plot(labels, hit10, label="Hit@10")
plt.plot(labels, mean_rank, label="Mean Rank")
plt.xlabel("Cosine Similarity Percentile")
plt.ylabel("Evaluation Metric")
# make y axis logarithmic
plt.yscale("log")
plt.title("Evaluation Metrics vs Cosine Similarity Percentile")
plt.legend()

plt.savefig(error_analysis_dir + "/cosine_percentile_eval.png")

print("===============================================")
print(error_analysis_dir)

percentile_lower_bound = np.min(cosine_similarity_score)
percentile_upper_bound = np.percentile(cosine_similarity_score, 80)

hit_at_1_average = np.mean(
    [
        hit_at_1[j]
        for j in range(len(hit_at_1))
        if percentile_lower_bound
        <= cosine_similarity_score[j]
        <= percentile_upper_bound
    ]
)
hit_at_5_average = np.mean(
    [
        hit_at_5[j]
        for j in range(len(hit_at_5))
        if percentile_lower_bound
        <= cosine_similarity_score[j]
        <= percentile_upper_bound
    ]
)
hit_at_10_average = np.mean(
    [
        hit_at_10[j]
        for j in range(len(hit_at_10))
        if percentile_lower_bound
        <= cosine_similarity_score[j]
        <= percentile_upper_bound
    ]
)
MR_average = np.mean(
    [
        MR[j]
        for j in range(len(MR))
        if percentile_lower_bound
        <= cosine_similarity_score[j]
        <= percentile_upper_bound
    ]
)

print(f"Percentile range: {percentile_lower_bound} - {percentile_upper_bound} (0-80%)")
print(f"Hit@1: {hit_at_1_average}")
print(f"Hit@5: {hit_at_5_average}")
print(f"Hit@10: {hit_at_10_average}")
print(f"Mean rank: {MR_average}")


percentile_lower_bound = np.percentile(cosine_similarity_score, 80)
percentile_upper_bound = np.max(cosine_similarity_score)

hit_at_1_average = np.mean(
    [
        hit_at_1[j]
        for j in range(len(hit_at_1))
        if percentile_lower_bound
        <= cosine_similarity_score[j]
        <= percentile_upper_bound
    ]
)
hit_at_5_average = np.mean(
    [
        hit_at_5[j]
        for j in range(len(hit_at_5))
        if percentile_lower_bound
        <= cosine_similarity_score[j]
        <= percentile_upper_bound
    ]
)
hit_at_10_average = np.mean(
    [
        hit_at_10[j]
        for j in range(len(hit_at_10))
        if percentile_lower_bound
        <= cosine_similarity_score[j]
        <= percentile_upper_bound
    ]
)
MR_average = np.mean(
    [
        MR[j]
        for j in range(len(MR))
        if percentile_lower_bound
        <= cosine_similarity_score[j]
        <= percentile_upper_bound
    ]
)

print(f"Percentile range: {percentile_lower_bound} - {percentile_upper_bound} (80-100%)")
print(f"Hit@1: {hit_at_1_average}")
print(f"Hit@5: {hit_at_5_average}")
print(f"Hit@10: {hit_at_10_average}")
print(f"Mean rank: {MR_average}")
