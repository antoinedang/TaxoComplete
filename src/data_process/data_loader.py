import dgl
import pickle
import os
import pandas as pd
import tqdm
import random
import networkx as nx
import data_process.helpers as helpers
import pdb
import datetime

MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000
date_time = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")


class Taxonx(nx.DiGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.root = [node for node in self.nodes() if self.in_degree(node) == 0]
        self.leaf_nodes = [node for node in self.nodes() if self.out_degree(node) == 0]


class TaxoDataset(object):
    def __init__(self, name, dir_path, raw=True, partition_pattern="leaf", seed=47):
        helpers.set_seed(seed)
        self.name = name  # taxonomy name
        self.partition_pattern = partition_pattern
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids
        self.test_node_ids = []  # a list of test node_ids
        if raw:
            self._load_dataset_raw(dir_path)
        else:
            self._load_dataset_pickled(dir_path)

    def _load_dataset_pickled(self, pickle_path):
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
        print("loading pickled data")
        self.name = data["name"]
        self.term2def = data["term2def"]
        self.taxonomy = data["taxonomy"]
        self.root = data["root"]
        self.leaf = data["leaf"]
        self.tx_id2name = data["tx_id2name"]
        self.tx_id2incrmt = data["tx_id2incrmt"]
        self.train_node_ids = data["train_node_ids"]
        self.validation_node_ids = data["validation_node_ids"]
        self.test_node_ids = data["test_node_ids"]

    def _load_dataset_raw(self, dir_path):
        node_file_name = os.path.join(dir_path, f"{self.name}.terms")
        def_file_name = os.path.join(dir_path, "term2def.csv")
        edge_file_name = os.path.join(dir_path, f"{self.name}.taxo")
        output_pickle_file_name = os.path.join(
            dir_path, f"{self.name}" + date_time + ".pickle.bin"
        )

        tx_id2name = {}
        tx_id2incrmt = {}
        # load nodes
        with open(node_file_name, "r") as fin:
            incr = 0
            for line in fin:
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    tx_id2name[segs[0]] = segs[1]
                    tx_id2incrmt[segs[0]] = incr
                    incr += 1
        self.tx_id2name = tx_id2name
        self.tx_id2incrmt = tx_id2incrmt
        # load edges
        tax_pairs = []
        with open(edge_file_name, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2incrmt[segs[0]]
                    child_taxon = tx_id2incrmt[segs[1]]
                    tax_pairs.append((parent_taxon, child_taxon))
        term2def = pd.read_csv(def_file_name)
        term2def = term2def.replace({"label": tx_id2incrmt})[["label", "summary"]]
        term2def.set_index("label")
        self.term2def = term2def.to_dict(orient="index")
        self.taxonomy = nx.DiGraph(tax_pairs)
        self.root = [
            node for node in self.taxonomy.nodes() if self.taxonomy.in_degree(node) == 0
        ]
        self.leaf = [
            node
            for node in self.taxonomy.nodes()
            if self.taxonomy.out_degree(node) == 0
        ]
        if self.partition_pattern == "leaf":
            random.shuffle(self.leaf)
            validation_size = min(int(len(self.leaf) * 0.1), MAX_VALIDATION_SIZE)
            test_size = min(int(len(self.leaf) * 0.1), MAX_TEST_SIZE)
            self.validation_node_ids = self.leaf[:validation_size]
            self.test_node_ids = self.leaf[
                validation_size : (validation_size + test_size)
            ]
            self.train_node_ids = [
                node_id
                for node_id in self.taxonomy.nodes
                if node_id not in self.validation_node_ids
                and node_id not in self.test_node_ids
            ]
        elif self.partition_pattern == "balanced":
            # balance leaf vs. non leaf nodes
            random.shuffle(self.leaf)
            num_leaf = len(self.leaf)
            num_non_leaf = len(self.taxonomy.nodes()) - num_leaf
            non_leaf = [node for node in self.taxonomy.nodes() if node not in self.leaf]
            if num_leaf > num_non_leaf:
                random_leaf_nodes = random.sample(
                    self.leaf,
                    num_non_leaf,
                )
                total_nodes = non_leaf + random_leaf_nodes
            else:
                random_nonleaf_nodes = random.sample(
                    non_leaf,
                    num_leaf,
                )
                total_nodes = self.leaf + random_nonleaf_nodes
            random.shuffle(total_nodes)
            validation_size = min(int(len(total_nodes) * 0.1), MAX_VALIDATION_SIZE)
            test_size = min(int(len(total_nodes) * 0.1), MAX_TEST_SIZE)
            self.validation_node_ids = total_nodes[:validation_size]
            self.test_node_ids = total_nodes[
                validation_size : (validation_size + test_size)
            ]
            self.train_node_ids = [
                node_id
                for node_id in total_nodes
                if node_id not in self.validation_node_ids
                and node_id not in self.test_node_ids
            ]
        elif self.partition_pattern == "balanced_test":
            # balance leaf vs. non leaf nodes (only in test and validation sets)
            random.shuffle(self.leaf)
            num_leaf = len(self.leaf)
            num_non_leaf = len(self.taxonomy.nodes()) - num_leaf
            non_leaf = [node for node in self.taxonomy.nodes() if node not in self.leaf]
            if num_leaf > num_non_leaf:
                random_leaf_nodes = random.sample(
                    self.leaf,
                    num_non_leaf,
                )
                total_nodes = non_leaf + random_leaf_nodes
            else:
                random_nonleaf_nodes = random.sample(
                    non_leaf,
                    num_leaf,
                )
                total_nodes = self.leaf + random_nonleaf_nodes
            random.shuffle(total_nodes)
            validation_size = min(int(len(total_nodes) * 0.1), MAX_VALIDATION_SIZE)
            test_size = min(int(len(total_nodes) * 0.1), MAX_TEST_SIZE)
            self.validation_node_ids = total_nodes[:validation_size]
            self.test_node_ids = total_nodes[
                validation_size : (validation_size + test_size)
            ]
            self.train_node_ids = [
                node_id
                for node_id in self.taxonomy.nodes
                if node_id not in self.validation_node_ids
                and node_id not in self.test_node_ids
            ]
        else:
            sampled_node_ids = [
                node for node in self.taxonomy.nodes() if node not in self.root
            ]
            random.shuffle(sampled_node_ids)

            validation_size = min(int(len(sampled_node_ids) * 0.1), MAX_VALIDATION_SIZE)
            test_size = min(int(len(sampled_node_ids) * 0.1), MAX_TEST_SIZE)
            self.validation_node_ids = sampled_node_ids[:validation_size]
            self.test_node_ids = sampled_node_ids[
                validation_size : (validation_size + test_size)
            ]
            self.train_node_ids = [
                node_id
                for node_id in self.taxonomy.nodes
                if node_id not in self.validation_node_ids
                and node_id not in self.test_node_ids
            ]
        # save to pickle for faster loading next time
        print("start saving pickle data")
        with open(output_pickle_file_name, "wb") as fout:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = {
                "name": self.name,
                "term2def": self.term2def,
                "taxonomy": self.taxonomy,
                "root": self.root,
                "leaf": self.leaf,
                "tx_id2name": self.tx_id2name,
                "tx_id2incrmt": self.tx_id2incrmt,
                "train_node_ids": self.train_node_ids,
                "validation_node_ids": self.validation_node_ids,
                "test_node_ids": self.test_node_ids,
            }
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")
