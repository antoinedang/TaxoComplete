import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

args = argparse.ArgumentParser(description="Visualize error analysis results.")
args.add_argument(
    type=str,
    help="Error analysis csv file path",
    dest="filename_1",
)
args.add_argument(
    type=str,
    help="Untrained error analysis csv file path",
    dest="filename_2",
)
args = args.parse_args()
# Define the file path to the CSV
csv_file_1_path = args.filename_1
csv_file_2_path = args.filename_2
log_dir_1 = os.path.dirname(csv_file_1_path)
log_dir_2 = os.path.dirname(csv_file_2_path)
log_filename_1 = log_dir_1 + "/embedding_comparison.log"
log_filename_2 = log_dir_2 + "/embedding_comparison.log"

show_plots = True

with open(log_filename_1, "w+") as f:
    f.write("== EMBEDDING COMPARISON LOG ==" + "\n")

with open(log_filename_2, "w+") as f:
    f.write("== EMBEDDING COMPARISON LOG ==" + "\n")

print_ = print


def print(*args):
    string = ""
    with open(log_filename_1, "a+", encoding="utf-8") as f1:
        with open(log_filename_2, "a+", encoding="utf-8") as f2:
            for var in args:
                string += str(var) + " "
            string = string[:-1]
            f1.write(string + "\n")
            f2.write(string + "\n")
    print_(string)


# Initialize separate lists for each column
queryDefs_1 = []
predChildDefs_1 = []
predParentDefs_1 = []
predChildPPRDefs_1 = []
predParentPPRDefs_1 = []
trueParentDefs_1 = []
trueChildDefs_1 = []
is_correct_parent_1 = []
is_correct_child_1 = []
is_correct_parent_ppr_1 = []
is_correct_child_ppr_1 = []
cos_sim_query_pred_child_1 = []
cos_sim_query_pred_parent_1 = []
cos_sim_query_pred_child_ppr_1 = []
cos_sim_query_pred_parent_ppr_1 = []
graph_distance_query_pred_child_1 = []
graph_distance_query_pred_parent_1 = []
graph_distance_query_pred_child_ppr_1 = []
graph_distance_query_pred_parent_ppr_1 = []

queryDefs_2 = []
predChildDefs_2 = []
predParentDefs_2 = []
predChildPPRDefs_2 = []
predParentPPRDefs_2 = []
trueParentDefs_2 = []
trueChildDefs_2 = []
is_correct_parent_2 = []
is_correct_child_2 = []
is_correct_parent_ppr_2 = []
is_correct_child_ppr_2 = []
cos_sim_query_pred_child_2 = []
cos_sim_query_pred_parent_2 = []
cos_sim_query_pred_child_ppr_2 = []
cos_sim_query_pred_parent_ppr_2 = []
graph_distance_query_pred_child_2 = []
graph_distance_query_pred_parent_2 = []
graph_distance_query_pred_child_ppr_2 = []
graph_distance_query_pred_parent_ppr_2 = []


def to_float_or_none(value):
    if value == "N/A":
        return np.nan
    return float(value)


# Open and read the CSV file
with open(csv_file_1_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    # Read each row and append values to respective lists
    for row in reader:
        queryDefs_1.append(row["queryDef"])
        predChildDefs_1.append(row["predChildDef"])
        predParentDefs_1.append(row["predParentDef"])
        predChildPPRDefs_1.append(row["predChildPPRDef"])
        predParentPPRDefs_1.append(row["predParentPPRDef"])
        trueParentDefs_1.append(row["true_parent_def"])
        trueChildDefs_1.append(row["true_child_def"])
        is_correct_parent_1.append(row["isCorrectParentAt1"] == "True")
        is_correct_child_1.append(row["isCorrectChildAt1"] == "True")
        is_correct_parent_ppr_1.append(row["isCorrectParentPPRAt1"] == "True")
        is_correct_child_ppr_1.append(row["isCorrectChildPPRAt1"] == "True")
        cos_sim_query_pred_child_1.append(
            to_float_or_none(row["cos_sim_query_pred_child"])
        )
        cos_sim_query_pred_parent_1.append(
            to_float_or_none(row["cos_sim_query_pred_parent"])
        )
        cos_sim_query_pred_child_ppr_1.append(
            to_float_or_none(row["cos_sim_query_pred_child_ppr"])
        )
        cos_sim_query_pred_parent_ppr_1.append(
            to_float_or_none(row["cos_sim_query_pred_parent_ppr"])
        )
        graph_distance_query_pred_child_1.append(
            to_float_or_none(row["graph_dist_query_pred_child"])
        )
        graph_distance_query_pred_parent_1.append(
            to_float_or_none(row["graph_dist_query_pred_parent"])
        )
        graph_distance_query_pred_child_ppr_1.append(
            to_float_or_none(row["graph_dist_query_pred_child_ppr"])
        )
        graph_distance_query_pred_parent_ppr_1.append(
            to_float_or_none(row["graph_dist_query_pred_parent_ppr"])
        )

# Open and read the CSV file
with open(csv_file_2_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    # Read each row and append values to respective lists
    for row in reader:
        queryDefs_2.append(row["queryDef"])
        predChildDefs_2.append(row["predChildDef"])
        predParentDefs_2.append(row["predParentDef"])
        predChildPPRDefs_2.append(row["predChildPPRDef"])
        predParentPPRDefs_2.append(row["predParentPPRDef"])
        trueParentDefs_2.append(row["true_parent_def"])
        trueChildDefs_2.append(row["true_child_def"])
        is_correct_parent_2.append(row["isCorrectParentAt1"] == "True")
        is_correct_child_2.append(row["isCorrectChildAt1"] == "True")
        is_correct_parent_ppr_2.append(row["isCorrectParentPPRAt1"] == "True")
        is_correct_child_ppr_2.append(row["isCorrectChildPPRAt1"] == "True")
        cos_sim_query_pred_child_2.append(
            to_float_or_none(row["cos_sim_query_pred_child"])
        )
        cos_sim_query_pred_parent_2.append(
            to_float_or_none(row["cos_sim_query_pred_parent"])
        )
        cos_sim_query_pred_child_ppr_2.append(
            to_float_or_none(row["cos_sim_query_pred_child_ppr"])
        )
        cos_sim_query_pred_parent_ppr_2.append(
            to_float_or_none(row["cos_sim_query_pred_parent_ppr"])
        )
        graph_distance_query_pred_child_2.append(
            to_float_or_none(row["graph_dist_query_pred_child"])
        )
        graph_distance_query_pred_parent_2.append(
            to_float_or_none(row["graph_dist_query_pred_parent"])
        )
        graph_distance_query_pred_child_ppr_2.append(
            to_float_or_none(row["graph_dist_query_pred_child_ppr"])
        )
        graph_distance_query_pred_parent_ppr_2.append(
            to_float_or_none(row["graph_dist_query_pred_parent_ppr"])
        )

is_correct_parent_1 = np.array(is_correct_parent_1)
is_correct_child_1 = np.array(is_correct_child_1)
is_correct_parent_ppr_1 = np.array(is_correct_parent_ppr_1)
is_correct_child_ppr_1 = np.array(is_correct_child_ppr_1)
cos_sim_query_pred_child_1 = np.array(cos_sim_query_pred_child_1)
cos_sim_query_pred_parent_1 = np.array(cos_sim_query_pred_parent_1)
cos_sim_query_pred_child_ppr_1 = np.array(cos_sim_query_pred_child_ppr_1)
cos_sim_query_pred_parent_ppr_1 = np.array(cos_sim_query_pred_parent_ppr_1)
graph_distance_query_pred_child_1 = np.array(graph_distance_query_pred_child_1)
graph_distance_query_pred_parent_1 = np.array(graph_distance_query_pred_parent_1)
graph_distance_query_pred_child_ppr_1 = np.array(graph_distance_query_pred_child_ppr_1)
graph_distance_query_pred_parent_ppr_1 = np.array(
    graph_distance_query_pred_parent_ppr_1
)
graph_distance_query_pred_parent_1[is_correct_parent_1] = 1
graph_distance_query_pred_child_1[is_correct_child_1] = 1
graph_distance_query_pred_parent_ppr_1[is_correct_parent_ppr_1] = 1
graph_distance_query_pred_child_ppr_1[is_correct_child_ppr_1] = 1

is_correct_parent_2 = np.array(is_correct_parent_2)
is_correct_child_2 = np.array(is_correct_child_2)
is_correct_parent_ppr_2 = np.array(is_correct_parent_ppr_2)
is_correct_child_ppr_2 = np.array(is_correct_child_ppr_2)
cos_sim_query_pred_child_2 = np.array(cos_sim_query_pred_child_2)
cos_sim_query_pred_parent_2 = np.array(cos_sim_query_pred_parent_2)
cos_sim_query_pred_child_ppr_2 = np.array(cos_sim_query_pred_child_ppr_2)
cos_sim_query_pred_parent_ppr_2 = np.array(cos_sim_query_pred_parent_ppr_2)
graph_distance_query_pred_child_2 = np.array(graph_distance_query_pred_child_2)
graph_distance_query_pred_parent_2 = np.array(graph_distance_query_pred_parent_2)
graph_distance_query_pred_child_ppr_2 = np.array(graph_distance_query_pred_child_ppr_2)
graph_distance_query_pred_parent_ppr_2 = np.array(
    graph_distance_query_pred_parent_ppr_2
)
graph_distance_query_pred_parent_2[is_correct_parent_2] = 1
graph_distance_query_pred_child_2[is_correct_child_2] = 1
graph_distance_query_pred_parent_ppr_2[is_correct_parent_ppr_2] = 1
graph_distance_query_pred_child_ppr_2[is_correct_child_ppr_2] = 1


identical_node_pairs_1 = set()
identical_node_pairs_1.update(
    [
        (queryDefs_1[idx], predParentDefs_1[idx])
        for idx in (
            np.where(
                (cos_sim_query_pred_parent_1 == 1)
                & (graph_distance_query_pred_parent_1 != 1)
            )[0]
        )
    ]
)
identical_node_pairs_1.update(
    [
        (queryDefs_1[idx], predChildDefs_1[idx])
        for idx in (
            np.where(
                (cos_sim_query_pred_child_1 == 1)
                & (graph_distance_query_pred_child_1 != 1)
            )[0]
        )
    ]
)
identical_node_pairs_1.update(
    [
        (queryDefs_1[idx], predParentPPRDefs_1[idx])
        for idx in (
            np.where(
                (cos_sim_query_pred_parent_ppr_1 == 1)
                & (graph_distance_query_pred_parent_ppr_1 != 1)
            )[0]
        )
    ]
)
identical_node_pairs_1.update(
    [
        (queryDefs_1[idx], predChildPPRDefs_1[idx])
        for idx in (
            np.where(
                (cos_sim_query_pred_child_ppr_1 == 1)
                & (graph_distance_query_pred_child_ppr_1 != 1)
            )[0]
        )
    ]
)

identical_node_pairs_2 = set()
identical_node_pairs_2.update(
    [
        (queryDefs_2[idx], predParentDefs_2[idx])
        for idx in (
            np.where(
                (cos_sim_query_pred_parent_2 == 1)
                & (graph_distance_query_pred_parent_2 != 1)
            )[0]
        )
    ]
)
identical_node_pairs_2.update(
    [
        (queryDefs_2[idx], predChildDefs_2[idx])
        for idx in (
            np.where(
                (cos_sim_query_pred_child_2 == 1)
                & (graph_distance_query_pred_child_2 != 1)
            )[0]
        )
    ]
)
identical_node_pairs_2.update(
    [
        (queryDefs_2[idx], predParentPPRDefs_2[idx])
        for idx in (
            np.where(
                (cos_sim_query_pred_parent_ppr_2 == 1)
                & (graph_distance_query_pred_parent_ppr_2 != 1)
            )[0]
        )
    ]
)
identical_node_pairs_2.update(
    [
        (queryDefs_2[idx], predChildPPRDefs_2[idx])
        for idx in (
            np.where(
                (cos_sim_query_pred_child_ppr_2 == 1)
                & (graph_distance_query_pred_child_ppr_2 != 1)
            )[0]
        )
    ]
)

print("=====================================")
print("Query/Node pairs with cosine similarity == 1:")
print("=============== BOTH ================")
intersection = identical_node_pairs_1.intersection(identical_node_pairs_2)
print(f"COUNT: {len(intersection)}")
for query, node in intersection:
    print(f"\nQuery:\n{query}\nNode:\n{node}")
print("=========== ONLY TRAINED ============")
subtraction = identical_node_pairs_1 - identical_node_pairs_2
print(f"COUNT: {len(subtraction)}")
for query, node in subtraction:
    print(f"\nQuery:\n{query}\nNode:\n{node}")
print("========== ONLY UNTRAINED ===========")
subtraction = identical_node_pairs_2 - identical_node_pairs_1
print(f"COUNT: {len(subtraction)}")
for query, node in subtraction:
    print(f"\nQuery:\n{query}\nNode:\n{node}")

print("=====================================")
print(f"TOTAL UNTRAINED: {len(identical_node_pairs_2)}")
print(f"TOTAL TRAINED: {len(identical_node_pairs_1)}")

print("==============AVG. ERROR=================")
non_nan_indices = ~np.isnan(cos_sim_query_pred_parent_1) & ~np.isnan(
    graph_distance_query_pred_parent_1
)
cos_sim_query_pred_parent_1 = cos_sim_query_pred_parent_1[non_nan_indices]
graph_distance_query_pred_parent_1 = graph_distance_query_pred_parent_1[non_nan_indices]
non_nan_indices = ~np.isnan(cos_sim_query_pred_child_1) & ~np.isnan(
    graph_distance_query_pred_child_1
)
cos_sim_query_pred_child_1 = cos_sim_query_pred_child_1[non_nan_indices]
graph_distance_query_pred_child_1 = graph_distance_query_pred_child_1[non_nan_indices]
non_nan_indices = ~np.isnan(cos_sim_query_pred_parent_ppr_1) & ~np.isnan(
    graph_distance_query_pred_parent_ppr_1
)
cos_sim_query_pred_parent_ppr_1 = cos_sim_query_pred_parent_ppr_1[non_nan_indices]
graph_distance_query_pred_parent_ppr_1 = graph_distance_query_pred_parent_ppr_1[
    non_nan_indices
]
non_nan_indices = ~np.isnan(cos_sim_query_pred_child_ppr_1) & ~np.isnan(
    graph_distance_query_pred_child_ppr_1
)
cos_sim_query_pred_child_ppr_1 = cos_sim_query_pred_child_ppr_1[non_nan_indices]
graph_distance_query_pred_child_ppr_1 = graph_distance_query_pred_child_ppr_1[
    non_nan_indices
]

non_nan_indices = ~np.isnan(cos_sim_query_pred_parent_2) & ~np.isnan(
    graph_distance_query_pred_parent_2
)
cos_sim_query_pred_parent_2 = cos_sim_query_pred_parent_2[non_nan_indices]
graph_distance_query_pred_parent_2 = graph_distance_query_pred_parent_2[non_nan_indices]
non_nan_indices = ~np.isnan(cos_sim_query_pred_child_2) & ~np.isnan(
    graph_distance_query_pred_child_2
)
cos_sim_query_pred_child_2 = cos_sim_query_pred_child_2[non_nan_indices]
graph_distance_query_pred_child_2 = graph_distance_query_pred_child_2[non_nan_indices]
non_nan_indices = ~np.isnan(cos_sim_query_pred_parent_ppr_2) & ~np.isnan(
    graph_distance_query_pred_parent_ppr_2
)
cos_sim_query_pred_parent_ppr_2 = cos_sim_query_pred_parent_ppr_2[non_nan_indices]
graph_distance_query_pred_parent_ppr_2 = graph_distance_query_pred_parent_ppr_2[
    non_nan_indices
]
non_nan_indices = ~np.isnan(cos_sim_query_pred_child_ppr_2) & ~np.isnan(
    graph_distance_query_pred_child_ppr_2
)
cos_sim_query_pred_child_ppr_2 = cos_sim_query_pred_child_ppr_2[non_nan_indices]
graph_distance_query_pred_child_ppr_2 = graph_distance_query_pred_child_ppr_2[
    non_nan_indices
]

graph_distance_query_pred_parent_1 = np.abs(graph_distance_query_pred_parent_1)
graph_distance_query_pred_child_1 = np.abs(graph_distance_query_pred_child_1)
graph_distance_query_pred_parent_ppr_1 = np.abs(graph_distance_query_pred_parent_ppr_1)
graph_distance_query_pred_child_ppr_1 = np.abs(graph_distance_query_pred_child_ppr_1)

graph_distance_query_pred_parent_2 = np.abs(graph_distance_query_pred_parent_2)
graph_distance_query_pred_child_2 = np.abs(graph_distance_query_pred_child_2)
graph_distance_query_pred_parent_ppr_2 = np.abs(graph_distance_query_pred_parent_ppr_2)
graph_distance_query_pred_child_ppr_2 = np.abs(graph_distance_query_pred_child_ppr_2)

error_cossim_graph_dist_1 = []
error_cossim_graph_dist_1.extend(
    (cos_sim_query_pred_parent_1 - (1 / graph_distance_query_pred_parent_1)).tolist()
)
error_cossim_graph_dist_1.extend(
    (cos_sim_query_pred_child_1 - (1 / graph_distance_query_pred_child_1)).tolist()
)
error_cossim_graph_dist_1.extend(
    (
        cos_sim_query_pred_parent_ppr_1 - (1 / graph_distance_query_pred_parent_ppr_1)
    ).tolist()
)
error_cossim_graph_dist_1.extend(
    (
        cos_sim_query_pred_child_ppr_1 - (1 / graph_distance_query_pred_child_ppr_1)
    ).tolist()
)

error_cossim_graph_dist_2 = []
error_cossim_graph_dist_2.extend(
    (cos_sim_query_pred_parent_2 - (1 / graph_distance_query_pred_parent_2)).tolist()
)
error_cossim_graph_dist_2.extend(
    (cos_sim_query_pred_child_2 - (1 / graph_distance_query_pred_child_2)).tolist()
)
error_cossim_graph_dist_2.extend(
    (
        cos_sim_query_pred_parent_ppr_2 - (1 / graph_distance_query_pred_parent_ppr_2)
    ).tolist()
)
error_cossim_graph_dist_2.extend(
    (
        cos_sim_query_pred_child_ppr_2 - (1 / graph_distance_query_pred_child_ppr_2)
    ).tolist()
)

print("============== TRAINED ==============")
mse_1 = np.mean(np.square(error_cossim_graph_dist_1))
print(f"MEAN SQUARED ERROR: {mse_1}")
print(f"STD. DEV: {np.std(error_cossim_graph_dist_1)}")

print("============ UNTRAINED =============")
mse_2 = np.mean(np.square(error_cossim_graph_dist_2))
print(f"MEAN SQUARED ERROR: {mse_2}")
print(f"STD. DEV: {np.std(error_cossim_graph_dist_2)}")
