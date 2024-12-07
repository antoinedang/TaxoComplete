import random
import numpy as np
import torch
from numbers import Number
from typing import Union
from pathlib import Path
import scipy.sparse as sp
from sentence_transformers import util

data_dir = Path(__file__).parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_cosine_ranges(range_percentile, query_embeddings, node_embeddings):
    cosine_similarities = util.cos_sim(query_embeddings, node_embeddings).cpu().numpy()
    cosine_similarities_sorted = np.sort(cosine_similarities)
    percentile_A = np.percentile(cosine_similarities_sorted, range_percentile[0])
    percentile_B = np.percentile(cosine_similarities_sorted, range_percentile[1])
    return [percentile_A, percentile_B]
