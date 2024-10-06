import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser(description="Visualize error analysis results.")
args.add_argument(
    type=str,
    help="Error analysis csv file path",
    dest="filename",
)
args = args.parse_args()
# Define the file path to the CSV
csv_file_path = args.filename

# Initialize separate lists for each column
queryDefs = []
predChildDefs = []
predParentDefs = []
predChildPPRDefs = []
predParentPPRDefs = []
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


def to_float_or_none(value):
    if value == "N/A":
        return None
    return float(value)


# Open and read the CSV file
with open(csv_file_path, mode="r") as file:
    reader = csv.DictReader(file)

    # Read each row and append values to respective lists
    for row in reader:
        queryDefs.append(row["queryDef"])
        predChildDefs.append(row["predChildDef"])
        predParentDefs.append(row["predParentDef"])
        predChildPPRDefs.append(row["predChildPPRDef"])
        predParentPPRDefs.append(row["predParentPPRDef"])
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
print(f"Overall Accuracy @ 1: {sum(correct_normal) / len(correct_normal)}")
correct_ppr = is_correct_parent_ppr & is_correct_child_ppr
print(f"Overall Accuracy @ 1 PPR: {sum(correct_ppr) / len(correct_ppr)}")


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
print(f"Overall Accuracy @ 10: {sum(correct_normal_10) / len(correct_normal_10)}")
correct_ppr_10 = is_correct_parent_ppr_10 & is_correct_child_ppr_10
print(f"Overall Accuracy @ 10 PPR: {sum(correct_ppr_10) / len(correct_ppr_10)}")

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
plt.ylabel("Accuracy @ 1")
plt.title("Accuracy @ 1 as a Function of Size of Close Neighborhood")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()


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
plt.ylabel("Accuracy @ 10")
plt.title("Accuracy @ 10 as a Function of Size of Close Neighborhood")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

has_graph_distance = graph_distance_query_pred_parent != None
inverse_graph_distance = 1 / (graph_distance_query_pred_parent)

correct_normal = is_correct_parent & has_graph_distance
not_correct_normal = ~is_correct_parent & has_graph_distance

plt.scatter(
    cos_sim_query_pred_parent[not_correct_normal],
    inverse_graph_distance[not_correct_normal],
    c="blue",
    label="Incorrect",
    # c=diff_to_cosine,
    # cmap="viridis",
    # alpha=0.5,
)
plt.scatter(
    cos_sim_query_pred_parent[correct_normal],
    inverse_graph_distance[correct_normal],
    c="red",
    label="Correct",
    # c=diff_to_cosine,
    # cmap="viridis",
    # alpha=0.5,
)
# plt.colorbar()
# plt.yscale("log")
plt.legend(["Incorrect", "Correct"])
plt.xlabel("Cosine Similarity")
plt.ylabel("Inverse Graph Distance")
plt.title("Cosine Similarity vs. Inverse Graph Distance")
plt.show()


has_graph_distance_ppr = graph_distance_query_pred_parent_ppr != None
inverse_graph_distance_ppr = 1 / (graph_distance_query_pred_parent_ppr)

correct_ppr = is_correct_parent_ppr & has_graph_distance_ppr
not_correct_ppr = ~is_correct_parent_ppr & has_graph_distance_ppr

plt.scatter(
    cos_sim_query_pred_parent_ppr[not_correct_ppr],
    inverse_graph_distance_ppr[not_correct_ppr],
    c="blue",
    label="Incorrect",
    # c=diff_to_cosine,
    # cmap="viridis",
    # alpha=0.5,
)
plt.scatter(
    cos_sim_query_pred_parent_ppr[correct_ppr],
    inverse_graph_distance_ppr[correct_ppr],
    c="red",
    label="Correct",
    # c=diff_to_cosine,
    # cmap="viridis",
    # alpha=0.5,
)
# plt.colorbar()
# plt.yscale("log")
plt.legend(["Incorrect", "Correct"])
plt.xlabel("Cosine Similarity")
plt.ylabel("Inverse Graph Distance")
plt.title("Cosine Similarity vs. Inverse Graph Distance (w/ PPR)")
plt.show()

print("=====================================")
print("Query/Node pairs with cosine similarity == 1:")
query_pred_parent_identical_embeddings = np.where(cos_sim_query_pred_parent == 1)[
    0
].tolist()
for idx in query_pred_parent_identical_embeddings:
    print(f"\nQuery:\n{queryDefs[idx]}\nPred. Parent:\n{predParentDefs[idx]}\n")


query_pred_child_identical_embeddings = np.where(cos_sim_query_pred_child == 1)[
    0
].tolist()
for idx in query_pred_child_identical_embeddings:
    print(f"\nQuery:\n{queryDefs[idx]}\nPred. Child:\n{predChildDefs[idx]}\n")


query_pred_parent_identical_embeddings_ppr = np.where(
    cos_sim_query_pred_parent_ppr == 1
)[0].tolist()
for idx in query_pred_parent_identical_embeddings_ppr:
    print(f"\nQuery:\n{queryDefs[idx]}\nPred. Parent PPR:\n{predParentPPRDefs[idx]}\n")


query_pred_child_identical_embeddings_ppr = np.where(cos_sim_query_pred_child_ppr == 1)[
    0
].tolist()
for idx in query_pred_child_identical_embeddings_ppr:
    print(f"\nQuery:\n{queryDefs[idx]}\nPred. Child PPR:\n{predChildPPRDefs[idx]}\n")
