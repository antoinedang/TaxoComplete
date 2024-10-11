import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

args = argparse.ArgumentParser(description="Visualize error analysis results.")
args.add_argument(
    type=str,
    help="Error analysis csv file path",
    dest="filename",
)
args = args.parse_args()
# Define the file path to the CSV
csv_file_path = args.filename
log_dir = os.path.dirname(csv_file_path)
log_filename = log_dir + "/error_analysis.log"

show_plots = True

with open(log_filename, "w+") as f:
    f.write("== ERROR ANALYSIS LOG ==" + "\n")

print_ = print


def print(*args):
    string = ""
    with open(log_filename, "a+", encoding="utf-8") as f:
        for var in args:
            string += str(var) + " "
        string = string[:-1]
        f.write(string + "\n")
    print_(string)


# Initialize separate lists for each column
queryDefs = []
predChildDefs = []
predParentDefs = []
predChildPPRDefs = []
predParentPPRDefs = []
trueParentDefs = []
trueChildDefs = []
size_of_close_neighborhood = []
query_level = []
query_height = []
is_correct_parent = []
is_correct_child = []
is_correct_parent_ppr = []
is_correct_child_ppr = []
is_correct_parent_10 = []
is_correct_child_10 = []
is_correct_parent_ppr_10 = []
is_correct_child_ppr_10 = []
cos_sim_query_pred_child = []
cos_sim_query_pred_parent = []
cos_sim_query_pred_child_ppr = []
cos_sim_query_pred_parent_ppr = []
graph_distance_query_pred_child = []
graph_distance_query_pred_parent = []
graph_distance_query_pred_child_ppr = []
graph_distance_query_pred_parent_ppr = []
relation_query_pred_parent = []
relation_query_pred_child = []
relation_query_pred_parent_ppr = []
relation_query_pred_child_ppr = []


def to_float_or_none(value):
    if value == "N/A":
        return None
    return float(value)


# Open and read the CSV file
with open(csv_file_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    # Read each row and append values to respective lists
    for row in reader:
        queryDefs.append(row["queryDef"])
        predChildDefs.append(row["predChildDef"])
        predParentDefs.append(row["predParentDef"])
        predChildPPRDefs.append(row["predChildPPRDef"])
        predParentPPRDefs.append(row["predParentPPRDef"])
        trueParentDefs.append(row["true_parent_def"])
        trueChildDefs.append(row["true_child_def"])
        size_of_close_neighborhood.append(int(row["numCloseNeighbors"]))
        query_level.append(int(row["queryLevel"]))
        query_height.append(int(row["queryHeight"]))
        is_correct_parent.append(row["isCorrectParentAt1"] == "True")
        is_correct_child.append(row["isCorrectChildAt1"] == "True")
        is_correct_parent_ppr.append(row["isCorrectParentPPRAt1"] == "True")
        is_correct_child_ppr.append(row["isCorrectChildPPRAt1"] == "True")
        is_correct_parent_10.append(row["isCorrectParentAt10"] == "True")
        is_correct_child_10.append(row["isCorrectChildAt10"] == "True")
        is_correct_parent_ppr_10.append(row["isCorrectParentPPRAt10"] == "True")
        is_correct_child_ppr_10.append(row["isCorrectChildPPRAt10"] == "True")
        cos_sim_query_pred_child.append(
            to_float_or_none(row["cos_sim_query_pred_child"])
        )
        cos_sim_query_pred_parent.append(
            to_float_or_none(row["cos_sim_query_pred_parent"])
        )
        cos_sim_query_pred_child_ppr.append(
            to_float_or_none(row["cos_sim_query_pred_child_ppr"])
        )
        cos_sim_query_pred_parent_ppr.append(
            to_float_or_none(row["cos_sim_query_pred_parent_ppr"])
        )
        graph_distance_query_pred_child.append(
            to_float_or_none(row["graph_dist_query_pred_child"])
        )
        graph_distance_query_pred_parent.append(
            to_float_or_none(row["graph_dist_query_pred_parent"])
        )
        graph_distance_query_pred_child_ppr.append(
            to_float_or_none(row["graph_dist_query_pred_child_ppr"])
        )
        graph_distance_query_pred_parent_ppr.append(
            to_float_or_none(row["graph_dist_query_pred_parent_ppr"])
        )
        relation_query_pred_parent.append(row["relation_query_pred_parent"])
        relation_query_pred_child.append(row["relation_query_pred_child"])
        relation_query_pred_parent_ppr.append(row["relation_query_pred_parent_ppr"])
        relation_query_pred_child_ppr.append(row["relation_query_pred_child_ppr"])

size_of_close_neighborhood = np.array(size_of_close_neighborhood)
query_level = np.array(query_level)
query_height = np.array(query_height)
is_correct_parent = np.array(is_correct_parent)
is_correct_child = np.array(is_correct_child)
is_correct_parent_ppr = np.array(is_correct_parent_ppr)
is_correct_child_ppr = np.array(is_correct_child_ppr)
is_correct_parent_10 = np.array(is_correct_parent_10)
is_correct_child_10 = np.array(is_correct_child_10)
is_correct_parent_ppr_10 = np.array(is_correct_parent_ppr_10)
is_correct_child_ppr_10 = np.array(is_correct_child_ppr_10)
cos_sim_query_pred_child = np.array(cos_sim_query_pred_child)
cos_sim_query_pred_parent = np.array(cos_sim_query_pred_parent)
cos_sim_query_pred_child_ppr = np.array(cos_sim_query_pred_child_ppr)
cos_sim_query_pred_parent_ppr = np.array(cos_sim_query_pred_parent_ppr)
graph_distance_query_pred_child = np.array(graph_distance_query_pred_child)
graph_distance_query_pred_parent = np.array(graph_distance_query_pred_parent)
graph_distance_query_pred_child_ppr = np.array(graph_distance_query_pred_child_ppr)
graph_distance_query_pred_parent_ppr = np.array(graph_distance_query_pred_parent_ppr)
relation_query_pred_parent = np.array(relation_query_pred_parent)
relation_query_pred_child = np.array(relation_query_pred_child)
relation_query_pred_parent_ppr = np.array(relation_query_pred_parent_ppr)
relation_query_pred_child_ppr = np.array(relation_query_pred_child_ppr)

graph_distance_query_pred_parent[is_correct_parent] = 1
graph_distance_query_pred_child[is_correct_child] = 1
graph_distance_query_pred_parent_ppr[is_correct_parent_ppr] = 1
graph_distance_query_pred_child_ppr[is_correct_child_ppr] = 1

# Get total counts for isCorrect columns
correctChild = sum(is_correct_child)
incorrectChild = len(is_correct_child) - correctChild
correctParent = sum(is_correct_parent)
incorrectParent = len(is_correct_parent) - correctParent
correctChildPPR = sum(is_correct_child_ppr)
incorrectChildPPR = len(is_correct_child_ppr) - correctChildPPR
correctParentPPR = sum(is_correct_parent_ppr)
incorrectParentPPR = len(is_correct_parent_ppr) - correctParentPPR

print(" >>>> HIT @ 1")
print("Correct Child: ", correctChild)
print("Incorrect Child: ", incorrectChild)
print("Correct Child PPR: ", correctChildPPR)
print("Incorrect Child PPR: ", incorrectChildPPR)
print("Correct Parent: ", correctParent)
print("Incorrect Parent: ", incorrectParent)
print("Correct Parent PPR: ", correctParentPPR)
print("Incorrect Parent PPR: ", incorrectParentPPR)

num_child_errors_introduced = 0
num_child_errors_rectified = 0
num_parent_errors_introduced = 0
num_parent_errors_rectified = 0
num_correct_parent_incorrect_child = 0
num_correct_parent_incorrect_child_ppr = 0
num_correct_child_incorrect_parent = 0
num_correct_child_incorrect_parent_ppr = 0

for i in range(len(is_correct_child)):
    if not is_correct_child[i] and is_correct_child_ppr[i]:
        num_child_errors_rectified += 1
    if is_correct_child[i] and not is_correct_child_ppr[i]:
        num_child_errors_introduced += 1
    if not is_correct_parent[i] and is_correct_parent_ppr[i]:
        num_parent_errors_rectified += 1
    if is_correct_parent[i] and not is_correct_parent_ppr[i]:
        num_parent_errors_introduced += 1

    if is_correct_parent[i] and not is_correct_child[i]:
        num_correct_parent_incorrect_child += 1
    if is_correct_parent_ppr[i] and not is_correct_child_ppr[i]:
        num_correct_parent_incorrect_child_ppr += 1
    if is_correct_child[i] and not is_correct_parent[i]:
        num_correct_child_incorrect_parent += 1
    if is_correct_child_ppr[i] and not is_correct_parent_ppr[i]:
        num_correct_child_incorrect_parent_ppr += 1

print("Num Child Errors Introduced: ", num_child_errors_introduced)
print("Num Child Errors Rectified: ", num_child_errors_rectified)
print("Num Parent Errors Introduced: ", num_parent_errors_introduced)
print("Num Parent Errors Rectified: ", num_parent_errors_rectified)
print("Num Correct Parent Incorrect Child: ", num_correct_parent_incorrect_child)
print(
    "Num Correct Parent Incorrect Child PPR: ",
    num_correct_parent_incorrect_child_ppr,
)
print("Num Correct Child Incorrect Parent: ", num_correct_child_incorrect_parent)
print(
    "Num Correct Child Incorrect Parent PPR: ",
    num_correct_child_incorrect_parent_ppr,
)

correct_normal = is_correct_parent & is_correct_child
print(f"Overall HIT @ 1: {sum(correct_normal) / len(correct_normal)}")
correct_ppr = is_correct_parent_ppr & is_correct_child_ppr
print(f"Overall HIT @ 1 PPR: {sum(correct_ppr) / len(correct_ppr)}")

is_leaf = query_height == 0
correct_normal_leaf = correct_normal[is_leaf]
print(
    f"Overall HIT @ 1 (Leaves): {sum(correct_normal_leaf) / len(correct_normal_leaf)}"
)
correct_ppr_leaf = correct_ppr[is_leaf]
print(f"Overall HIT @ 1 PPR (Leaves): {sum(correct_ppr_leaf) / len(correct_ppr_leaf)}")

correct_normal_non_leaf = correct_normal[~is_leaf]
print(
    f"Overall HIT @ 1 (Non-Leaves): {sum(correct_normal_non_leaf) / len(correct_normal_non_leaf)}"
)
correct_ppr_non_leaf = correct_ppr[~is_leaf]
print(
    f"Overall HIT @ 1 PPR (Non-Leaves): {sum(correct_ppr_non_leaf) / len(correct_ppr_non_leaf)}"
)


# Get total counts for isCorrect columns
correctChild_10 = sum(is_correct_child_10)
incorrectChild_10 = len(is_correct_child_10) - correctChild_10
correctParent_10 = sum(is_correct_parent_10)
incorrectParent_10 = len(is_correct_parent_10) - correctParent_10
correctChildPPR_10 = sum(is_correct_child_ppr_10)
incorrectChildPPR_10 = len(is_correct_child_ppr_10) - correctChildPPR_10
correctParentPPR_10 = sum(is_correct_parent_ppr_10)
incorrectParentPPR_10 = len(is_correct_parent_ppr_10) - correctParentPPR_10

print(" >>>> HIT @ 10")
print("Correct Child: ", correctChild_10)
print("Incorrect Child: ", incorrectChild_10)
print("Correct Child PPR: ", correctChildPPR_10)
print("Incorrect Child PPR: ", incorrectChildPPR_10)
print("Correct Parent: ", correctParent_10)
print("Incorrect Parent: ", incorrectParent_10)
print("Correct Parent PPR: ", correctParentPPR_10)
print("Incorrect Parent PPR: ", incorrectParentPPR_10)

num_child_errors_introduced_10 = 0
num_child_errors_rectified_10 = 0
num_parent_errors_introduced_10 = 0
num_parent_errors_rectified_10 = 0
num_correct_parent_incorrect_child_10 = 0
num_correct_parent_incorrect_child_10_ppr = 0
num_correct_child_incorrect_parent_10 = 0
num_correct_child_incorrect_parent_10_ppr = 0

for i in range(len(is_correct_child_10)):
    if not is_correct_child_10[i] and is_correct_child_ppr_10[i]:
        num_child_errors_rectified_10 += 1
    if is_correct_child_10[i] and not is_correct_child_ppr_10[i]:
        num_child_errors_introduced_10 += 1
    if not is_correct_parent_10[i] and is_correct_parent_ppr_10[i]:
        num_parent_errors_rectified_10 += 1
    if is_correct_parent_10[i] and not is_correct_parent_ppr_10[i]:
        num_parent_errors_introduced_10 += 1

    if is_correct_parent_10[i] and not is_correct_child_10[i]:
        num_correct_parent_incorrect_child_10 += 1
    if is_correct_parent_ppr_10[i] and not is_correct_child_ppr_10[i]:
        num_correct_parent_incorrect_child_10_ppr += 1
    if is_correct_child_10[i] and not is_correct_parent_10[i]:
        num_correct_child_incorrect_parent_10 += 1
    if is_correct_child_ppr_10[i] and not is_correct_parent_ppr_10[i]:
        num_correct_child_incorrect_parent_10_ppr += 1

print("Num Child Errors Introduced: ", num_child_errors_introduced_10)
print("Num Child Errors Rectified: ", num_child_errors_rectified_10)
print("Num Parent Errors Introduced: ", num_parent_errors_introduced_10)
print("Num Parent Errors Rectified: ", num_parent_errors_rectified_10)
print("Num Correct Parent Incorrect Child: ", num_correct_parent_incorrect_child_10)
print(
    "Num Correct Parent Incorrect Child PPR: ",
    num_correct_parent_incorrect_child_10_ppr,
)
print("Num Correct Child Incorrect Parent: ", num_correct_child_incorrect_parent_10)
print(
    "Num Correct Child Incorrect Parent PPR: ",
    num_correct_child_incorrect_parent_10_ppr,
)

correct_normal_10 = is_correct_parent_10 & is_correct_child_10
print(f"Overall HIT @ 10: {sum(correct_normal_10) / len(correct_normal_10)}")
correct_ppr_10 = is_correct_parent_ppr_10 & is_correct_child_ppr_10
print(f"Overall HIT @ 10 PPR: {sum(correct_ppr_10) / len(correct_ppr_10)}")

is_leaf = query_height == 0
correct_normal_leaf = correct_normal_10[is_leaf]
print(
    f"Overall HIT @ 10 (Leaves): {sum(correct_normal_leaf) / len(correct_normal_leaf)}"
)
correct_ppr_leaf = correct_ppr_10[is_leaf]
print(f"Overall HIT @ 10 PPR (Leaves): {sum(correct_ppr_leaf) / len(correct_ppr_leaf)}")

correct_normal_non_leaf = correct_normal_10[~is_leaf]
print(
    f"Overall HIT @ 10 (Non-Leaves): {sum(correct_normal_non_leaf) / len(correct_normal_non_leaf)}"
)
correct_ppr_non_leaf = correct_ppr_10[~is_leaf]
print(
    f"Overall HIT @ 10 PPR (Non-Leaves): {sum(correct_ppr_non_leaf) / len(correct_ppr_non_leaf)}"
)

num_leaf_queries = 0
for i in range(len(query_height)):
    if query_height[i] == 0:
        num_leaf_queries += 1

print("Num Leaf Queries: ", num_leaf_queries)
print("Num Non-Leaf Queries: ", len(query_height) - num_leaf_queries)

num_bins = 10
sorted_indices = np.argsort(size_of_close_neighborhood)
sorted_sizes = size_of_close_neighborhood[sorted_indices]
correct_normal = correct_normal[sorted_indices]
correct_ppr = correct_ppr[sorted_indices]

bin_size = len(size_of_close_neighborhood) // num_bins

accuracy_normal = []
accuracy_ppr = []
bucket_centers = []
bucket_ranges = []

for i in range(num_bins):
    start_idx = i * bin_size
    if i == num_bins - 1:
        end_idx = len(size_of_close_neighborhood)
    else:
        end_idx = (i + 1) * bin_size

    bin_sizes = sorted_sizes[start_idx:end_idx]
    bin_correct_normal = correct_normal[start_idx:end_idx]
    bin_correct_ppr = correct_ppr[start_idx:end_idx]

    accuracy_normal.append(np.mean(bin_correct_normal))
    accuracy_ppr.append(np.mean(bin_correct_ppr))
    bucket_centers.append(np.mean(bin_sizes))
    bucket_ranges.append((bin_sizes[0], bin_sizes[-1]))

bar_width = 0.35
x = np.arange(len(bucket_centers))
bars1 = plt.bar(
    x - bar_width / 2, accuracy_normal, bar_width, label="Normal Predictions"
)
bars2 = plt.bar(x + bar_width / 2, accuracy_ppr, bar_width, label="PPR Predictions")
plt.xticks(x, [f"{int(range[0])} - {int(range[1])}" for range in bucket_ranges])
plt.xlabel("Size of Close Neighborhood")
plt.ylabel("HIT @ 1")
plt.title("HIT @ 1 as a Function of Size of Close Neighborhood")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(log_dir + "/neighborhood_accuracy.png")
if show_plots:
    plt.show()
plt.clf()

correct_normal_10 = correct_normal_10[sorted_indices]
correct_ppr_10 = correct_ppr_10[sorted_indices]

bin_size = len(size_of_close_neighborhood) // num_bins

accuracy_normal_10 = []
accuracy_ppr_10 = []
bucket_centers = []
bucket_ranges = []

for i in range(num_bins):
    start_idx = i * bin_size
    if i == num_bins - 1:
        end_idx = len(size_of_close_neighborhood)
    else:
        end_idx = (i + 1) * bin_size

    bin_sizes = sorted_sizes[start_idx:end_idx]
    bin_correct_normal_10 = correct_normal_10[start_idx:end_idx]
    bin_correct_ppr_10 = correct_ppr_10[start_idx:end_idx]

    accuracy_normal_10.append(np.mean(bin_correct_normal_10))
    accuracy_ppr_10.append(np.mean(bin_correct_ppr_10))
    bucket_centers.append(np.mean(bin_sizes))
    bucket_ranges.append((bin_sizes[0], bin_sizes[-1]))

bar_width = 0.35
x = np.arange(len(bucket_centers))
bars1 = plt.bar(
    x - bar_width / 2, accuracy_normal_10, bar_width, label="Normal Predictions"
)
bars2 = plt.bar(x + bar_width / 2, accuracy_ppr_10, bar_width, label="PPR Predictions")
plt.xticks(x, [f"{int(range[0])} - {int(range[1])}" for range in bucket_ranges])
plt.xlabel("Size of Close Neighborhood")
plt.ylabel("HIT @ 10")
plt.title("HIT @ 10 as a Function of Size of Close Neighborhood")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(log_dir + "/neighborhood_accuracy_10.png")
if show_plots:
    plt.show()
plt.clf()


def plot_dist_vs_cossim(
    graph_dist, cossim, is_correct, title, filename, relations, ignore_relation
):
    has_graph_distance = graph_dist != None
    has_cos_sim = cossim != None
    correct = is_correct & has_graph_distance & has_cos_sim
    not_correct = ~is_correct & has_graph_distance & has_cos_sim

    not_correct_child = not_correct & (relations == "child")
    not_correct_parent = not_correct & (relations == "parent")
    not_correct_sibling = not_correct & (relations == "sibling")
    not_correct_ancestor = not_correct & (relations == "ancestor")
    not_correct_descendant = not_correct & (relations == "descendant")
    not_correct_distant = not_correct & (relations == "distant")

    for color, label, vals in zip(
        ["yellow", "green", "purple", "orange", "brown", "blue"],
        [
            "Incorrect Child",
            "Incorrect Parent",
            "Incorrect Sibling",
            "Incorrect Ancestor",
            "Incorrect Descendant",
            "Incorrect",
        ],
        [
            not_correct_child,
            not_correct_parent,
            not_correct_sibling,
            not_correct_ancestor,
            not_correct_descendant,
            not_correct_distant,
        ],
    ):
        if sum(vals) == 0:
            continue
        if label == ignore_relation:
            continue
        inverse_graph_distance_not_correct = 1 / (graph_dist[vals])

        plt.scatter(
            cossim[vals],
            inverse_graph_distance_not_correct,
            c=color,
            label=label,
        )
    inverse_graph_distance_correct = 1 / (graph_dist[correct])
    plt.scatter(
        cossim[correct],
        inverse_graph_distance_correct,
        c="red",
        label="Correct",
    )
    plt.legend()
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Inverse Graph Distance")
    plt.title(title)
    plt.savefig(log_dir + "/" + filename)
    if show_plots:
        plt.show()

    plt.clf()


plot_dist_vs_cossim(
    graph_distance_query_pred_child,
    cos_sim_query_pred_child,
    is_correct_child,
    "Cosine Similarity vs. Inverse Graph Distance (Child, no PPR)",
    "dist_vs_cossim_child.png",
    relation_query_pred_child,
    "Incorrect Child",
)
plot_dist_vs_cossim(
    graph_distance_query_pred_parent,
    cos_sim_query_pred_parent,
    is_correct_parent,
    "Cosine Similarity vs. Inverse Graph Distance (Parent, no PPR)",
    "dist_vs_cossim_parent.png",
    relation_query_pred_parent,
    "Incorrect Parent",
)
plot_dist_vs_cossim(
    graph_distance_query_pred_child_ppr,
    cos_sim_query_pred_child_ppr,
    is_correct_child_ppr,
    "Cosine Similarity vs. Inverse Graph Distance (Child, PPR)",
    "dist_vs_cossim_child_ppr.png",
    relation_query_pred_child_ppr,
    "Incorrect Child",
)
plot_dist_vs_cossim(
    graph_distance_query_pred_parent_ppr,
    cos_sim_query_pred_parent_ppr,
    is_correct_parent_ppr,
    "Cosine Similarity vs. Inverse Graph Distance (Parent, PPR)",
    "dist_vs_cossim_parent_ppr.png",
    relation_query_pred_parent_ppr,
    "Incorrect Parent",
)


def print_identical_embedding_definitions(
    query_defs, node_defs, true_defs, cossims, isPPR, isChild, correct
):
    ppr_text = " PPR" if isPPR else ""
    relation_text = "Child" if isChild else "Parent"
    identical_embeddings = np.where(cossims == 1)[0].tolist()
    for idx in identical_embeddings:
        if not correct[idx]:
            print(
                f"\nQuery:\n{query_defs[idx]}\nPred. {relation_text}{ppr_text}:\n{node_defs[idx]}\nTrue {relation_text}:\n{true_defs[idx]}"
            )


print("=====================================")
print("Query/Node pairs with cosine similarity == 1:")

print_identical_embedding_definitions(
    queryDefs,
    predParentDefs,
    trueParentDefs,
    cos_sim_query_pred_parent,
    False,
    False,
    is_correct_parent,
)
print_identical_embedding_definitions(
    queryDefs,
    predChildDefs,
    trueChildDefs,
    cos_sim_query_pred_child,
    False,
    True,
    is_correct_child,
)
print_identical_embedding_definitions(
    queryDefs,
    predParentPPRDefs,
    trueParentDefs,
    cos_sim_query_pred_parent_ppr,
    True,
    False,
    is_correct_parent_ppr,
)
print_identical_embedding_definitions(
    queryDefs,
    predChildPPRDefs,
    trueChildDefs,
    cos_sim_query_pred_child_ppr,
    True,
    True,
    is_correct_child_ppr,
)
print("=====================================")

print("== CSV FOR ONLY PARENTS WITH COS_SIM @ 1 ==" + "\n")
print("query def, true parent def, pred parent def ppr, pred parent def")


def log_identical_embedding_definitions(
    query_defs, node_defs, true_defs, cossims, isPPR, correct
):
    identical_embeddings = np.where(cossims == 1)[0].tolist()
    for idx in identical_embeddings:
        if not correct[idx]:
            print(
                f'"{query_defs[idx]}","{true_defs[idx]}","{"N/A" if not isPPR else node_defs[idx]}","{"N/A" if isPPR else node_defs[idx]}"'
            )


log_identical_embedding_definitions(
    queryDefs,
    predParentDefs,
    trueParentDefs,
    cos_sim_query_pred_parent,
    False,
    is_correct_parent,
)
log_identical_embedding_definitions(
    queryDefs,
    predParentPPRDefs,
    trueParentDefs,
    cos_sim_query_pred_parent_ppr,
    True,
    is_correct_parent_ppr,
)
