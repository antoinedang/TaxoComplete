import torch
from torch import nn, Tensor
from typing import Iterable, Dict, List
import numpy as np
import math
from scipy.special import lambertw


def exp_map_hyperboloid(x, c=1.0):
    if type(x) is np.ndarray:
        norm_x = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x_hyperboloid = np.concatenate([np.sqrt(c + norm_x**2), x], axis=-1)
        return x_hyperboloid
    norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_hyperboloid = torch.cat([torch.sqrt(c + norm_x**2), x], dim=-1)
    return x_hyperboloid


def lorentzian_inner_product(u, v, c=1.0):
    try:
        u_dim = u.dim()
        sum_fn = torch.sum
    except AttributeError:
        u_dim = u.ndim

        def sum_fn(x, dim=None):
            if dim is None:
                return np.sum(x)
            else:
                return np.sum(x, axis=dim)

    if u_dim == 1:  # Non-batch case
        return -(c**2) * u[0] * v[0] + sum_fn(u[1:] * v[1:])
    else:  # Batch-wise case
        return -(c**2) * u[:, 0] * v[:, 0] + sum_fn(u[:, 1:] * v[:, 1:], dim=-1)


def lorentz_norm(u, c=1.0):
    if type(u) is np.ndarray:
        return np.sqrt(
            np.clip(-lorentzian_inner_product(u, u, c), a_min=1e-5, a_max=None)
        )
    inner_prod = lorentzian_inner_product(u, u, c)
    return torch.sqrt(torch.clamp(-inner_prod, min=1e-5))


def hyperbolic_cosine_similarity(u, v, c=1.0):
    inner_prod = lorentzian_inner_product(u, v, c)
    norm_u = lorentz_norm(u, c)
    norm_v = lorentz_norm(v, c)
    return inner_prod / (norm_u * norm_v)


class CosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    Example::

            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)


    """

    def __init__(
        self,
        model,
        loss_fct=nn.MSELoss(),
        alpha=1,
        beta=0,
        modified_loss=False,
        hyperbolic=False,
        hyperbolic_curvature=1.0,
        cosine_absolute=False,
        super_loss=False,
        super_loss_tau=math.log(10),
        super_loss_lam=1.0,
        super_loss_fac=0.0,
    ):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.alpha = alpha
        self.beta = beta
        self.modified_loss = modified_loss
        self.hyperbolic = hyperbolic
        self.hyperbolic_curvature = hyperbolic_curvature
        self.cosine_absolute = cosine_absolute
        self.super_loss = super_loss
        self.tau = super_loss_tau
        self.lam = super_loss_lam
        self.fac = super_loss_fac

    def _super_loss(self, loss):
        print("loss 2", loss)
        print("loss detached", loss.detach())
        print("loss detached cpu", loss.detach().cpu())
        origin_loss = loss.detach().cpu().numpy()
        print("origin_loss", origin_loss)
        if self.fac > 0.0:
            self.tau = self.fac * origin_loss.mean() + (1.0 - self.fac) * self.tau

        beta = (origin_loss - self.tau) / self.lam
        gamma = -2.0 / np.exp(1.0)
        sigma = np.exp(-lambertw(0.5 * np.maximum(beta, gamma))).real
        sigma = torch.from_numpy(np.array(sigma))  # .to(self.device)
        super_loss = (loss - self.tau) * sigma + self.lam * (torch.log(sigma) ** 2)
        return torch.mean(super_loss)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: List):
        loss = self.__forward(sentence_features, labels)
        if self.super_loss:
            print("loss 1", loss)
            return self._super_loss(loss)
        return loss

    def __forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: List):
        # pdb.set_trace()
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        query_embedding = embeddings[0]
        corpus_embedding = embeddings[1]
        parent_embedding = embeddings[2]

        if self.hyperbolic:
            query_embedding = exp_map_hyperboloid(
                query_embedding, self.hyperbolic_curvature
            )
            corpus_embedding = exp_map_hyperboloid(
                corpus_embedding, self.hyperbolic_curvature
            )
            parent_embedding = exp_map_hyperboloid(
                parent_embedding, self.hyperbolic_curvature
            )
            similarity_measure = lambda x, y: hyperbolic_cosine_similarity(
                x, y, self.hyperbolic_curvature
            )
        else:
            similarity_measure = torch.cosine_similarity

        if self.cosine_absolute:
            similarity_measure_ = lambda x, y: torch.abs(
                similarity_measure(x, y)
            ).float()
        else:
            similarity_measure_ = similarity_measure

        if self.modified_loss:
            query_corpus_cossim = similarity_measure_(query_embedding, corpus_embedding)
            return self.loss_fct(
                float(-2) * query_corpus_cossim + float(3),
                labels[:, 0].view(-1).float(),
            ).float()
        else:
            query_corpus_loss = similarity_measure_(query_embedding, corpus_embedding)
            query_parent_loss = similarity_measure_(query_embedding, parent_embedding)

            return (
                self.alpha
                * self.loss_fct(
                    query_corpus_loss, labels[:, 0].view(-1).float()
                ).float()
                + self.beta
                * self.loss_fct(
                    query_parent_loss, torch.ones_like(labels[:, 0].view(-1)).float()
                ).float()
            )
