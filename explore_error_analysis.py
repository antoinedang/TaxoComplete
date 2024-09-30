import csv
import numpy as np
import matplotlib.pyplot as plt

# Define the file path to the CSV
csv_file_path = "error_analysis.csv"

# Initialize separate lists for each column
size_of_close_neighborhood = []
query_level = []
query_height = []
is_correct_parent = []
is_correct_child = []
is_correct_parent_ppr = []
is_correct_child_ppr = []
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
        size_of_close_neighborhood.append(int(row["sizeOfCloseNeighborhood"]))
        query_level.append(int(row["queryLevel"]))
        query_height.append(int(row["queryHeight"]))
        is_correct_parent.append(row["isCorrectParent"] == "True")
        is_correct_child.append(row["isCorrectChild"] == "True")
        is_correct_parent_ppr.append(row["isCorrectParentPPR"] == "True")
        is_correct_child_ppr.append(row["isCorrectChildPPR"] == "True")
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
            to_float_or_none(row["graph_distance_query_pred_child"])
        )
        graph_distance_query_pred_parent.append(
            to_float_or_none(row["graph_distance_query_pred_parent"])
        )
        graph_distance_query_pred_child_ppr.append(
            to_float_or_none(row["graph_distance_query_pred_child_ppr"])
        )
        graph_distance_query_pred_parent_ppr.append(
            to_float_or_none(row["graph_distance_query_pred_parent_ppr"])
        )

size_of_close_neighborhood = np.array(size_of_close_neighborhood)
query_level = np.array(query_level)
query_height = np.array(query_height)
is_correct_parent = np.array(is_correct_parent)
is_correct_child = np.array(is_correct_child)
is_correct_parent_ppr = np.array(is_correct_parent_ppr)
is_correct_child_ppr = np.array(is_correct_child_ppr)
cos_sim_query_pred_child = np.array(cos_sim_query_pred_child)
cos_sim_query_pred_parent = np.array(cos_sim_query_pred_parent)
cos_sim_query_pred_child_ppr = np.array(cos_sim_query_pred_child_ppr)
cos_sim_query_pred_parent_ppr = np.array(cos_sim_query_pred_parent_ppr)
graph_distance_query_pred_child = np.array(graph_distance_query_pred_child)
graph_distance_query_pred_parent = np.array(graph_distance_query_pred_parent)
graph_distance_query_pred_child_ppr = np.array(graph_distance_query_pred_child_ppr)
graph_distance_query_pred_parent_ppr = np.array(graph_distance_query_pred_parent_ppr)

# Get total counts for isCorrect columns
correctChild = sum(is_correct_child)
incorrectChild = len(is_correct_child) - correctChild
correctParent = sum(is_correct_parent)
incorrectParent = len(is_correct_parent) - correctParent
correctChildPPR = sum(is_correct_child_ppr)
incorrectChildPPR = len(is_correct_child_ppr) - correctChildPPR
correctParentPPR = sum(is_correct_parent_ppr)
incorrectParentPPR = len(is_correct_parent_ppr) - correctParentPPR

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

for i in range(len(is_correct_child)):
    if not is_correct_child[i] and is_correct_child_ppr[i]:
        num_child_errors_rectified += 1
    if is_correct_child[i] and not is_correct_child_ppr[i]:
        num_child_errors_introduced += 1
    if not is_correct_parent[i] and is_correct_parent_ppr[i]:
        num_parent_errors_rectified += 1
    if is_correct_parent[i] and not is_correct_parent_ppr[i]:
        num_parent_errors_introduced += 1

print("Num Child Errors Introduced: ", num_child_errors_introduced)
print("Num Child Errors Rectified: ", num_child_errors_rectified)
print("Num Parent Errors Introduced: ", num_parent_errors_introduced)
print("Num Parent Errors Rectified: ", num_parent_errors_rectified)

num_leaf_queries = 0
for i in range(len(query_height)):
    if query_height[i] == 0:
        num_leaf_queries += 1

print("Num Leaf Queries: ", num_leaf_queries)
print("Num Non-Leaf Queries: ", len(query_height) - num_leaf_queries)


num_bins = 10
correct_normal = is_correct_parent & is_correct_child
correct_ppr = is_correct_parent_ppr & is_correct_child_ppr

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
plt.ylabel("Accuracy")
plt.title("Accuracy as a Function of Size of Close Neighborhood")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
