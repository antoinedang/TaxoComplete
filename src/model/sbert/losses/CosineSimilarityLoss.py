import torch
from torch import nn, Tensor
from typing import Iterable, Dict, List
from ..SentenceTransformer import SentenceTransformer
import pdb


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
        model: SentenceTransformer,
        loss_fct=nn.MSELoss(),
        alpha=1,
        beta=0,
        modified_loss=False,
    ):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.alpha = alpha
        self.beta = beta
        self.modified_loss = modified_loss

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: List):
        # pdb.set_trace()
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        query_embedding = embeddings[0]
        corpus_embedding = embeddings[1]
        parent_embedding = embeddings[2]

        if self.modified_loss:
            query_corpus_cossim = torch.cosine_similarity(
                query_embedding, corpus_embedding
            )
            return (-2 * query_corpus_cossim + 3 - labels[:, 0].view(-1)) / len(
                labels[:, 0].view(-1)
            )
        else:
            query_corpus_loss = torch.cosine_similarity(
                query_embedding, corpus_embedding
            )
            query_parent_loss = torch.cosine_similarity(
                query_embedding, parent_embedding
            )

            return self.alpha * self.loss_fct(
                query_corpus_loss, labels[:, 0].view(-1)
            ) + self.beta * self.loss_fct(
                query_parent_loss, torch.ones_like(labels[:, 0].view(-1))
            )
