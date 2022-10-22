from typing import List, Dict, Iterable, Any, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity


from emorec.emorec_utils.trainer import (
    SemEval2018Task1EcTrainer,
    GoEmotionsTrainer,
    FrenchElectionTrainer,
)
from emorec.train_utils import Correlations
from emorec.utils import flatten_list


class DemuxTrainerMixin:
    """Demux dataset mixin.

    Attributes:
        global_correlations: `Correlations` module to get weights for
            each pair of emotions in a global loss.
    """

    argparse_args = dict(
        global_correlation_coef=dict(
            type=float, help="Global correlation loss coefficient"
        ),
        global_correlation_loss=dict(
            type=str,
            help="what global loss function to use",
            default="cossim",
        ),
        global_priors=dict(
            action="store_true",
            help="whether to use prior correlations rather than "
            "data-driven ones",
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dataset_labels = self.dataset.labels
        if isinstance(dataset_labels, list):
            dataset_labels = torch.cat(dataset_labels)
        self.global_correlations = Correlations(
            dataset_labels if not self.exp_handler.global_priors else None,
            self.dataset.all_emotions,
            active=self.exp_handler.global_correlation_coef is not None,
            normalize=self.exp_handler.global_correlation_loss == "sq_diff",
        )

    _global_distances = dict(
        cossim=lambda repr, corrs: (
            cosine_similarity(
                repr.unsqueeze(-1),
                repr.unsqueeze(1).transpose(2, 3),
                dim=2,
            )
            - corrs
        )
        .triu(diagonal=1)
        .square()
        .mean(),
        sq_diff=lambda repr, corrs: (repr.unsqueeze(2) - repr.unsqueeze(1))
        .norm(dim=-1)
        .triu(diagonal=1)
        .mul(corrs)
        .mean(),
    )

    def calculate_regularization_loss(
        self,
        intermediate_representations: Optional[torch.Tensor],
        logits: torch.Tensor,
        batch: Iterable[Any],
        train: bool,
    ) -> torch.Tensor:

        loss = super().calculate_regularization_loss(
            intermediate_representations, logits, batch, train
        )

        if train:
            emotions = np.array(self.dataset.emotions, dtype=object)
        else:
            emotions = np.array(self.dataset.all_emotions, dtype=object)

        if self.exp_handler.global_correlation_coef:
            corrs = self.global_correlations.get(
                (emotions.tolist(), emotions.tolist()), decreasing=False
            ).to(self.exp_handler.device)

            global_loss = self._global_distances[
                self.exp_handler.global_correlation_loss
            ](intermediate_representations, corrs)

            loss = loss + self.exp_handler.global_correlation_coef * global_loss

        return loss

    def get_logits_from_model(
        self,
        return_vals: Any,
        batch: Iterable[Any],
        data_loader: DataLoader,
        epoch: int = -1,
    ) -> torch.Tensor:
        return return_vals[0]

    def get_intermediate_repr_from_model(
        self, return_vals: Any, batch: Iterable[Any]
    ) -> Optional[torch.Tensor]:
        return return_vals[1]

    def eval_init(self, data_loader: DataLoader):
        self.model.set_class_inds(data_loader.dataset.class_inds)
        return super().eval_init(data_loader)

    def eval_end(self, data_loader: DataLoader):
        self.model.set_class_inds(self.dataset.class_inds)
        return super().eval_end(data_loader)

    def train_init(self):
        super().train_init()
        if self.exp_handler.freeze_word_embeddings:
            self.model.freeze_word_embeddings()
        if self.exp_handler.freeze_emotion_embeddings:
            self._emotion_embeddings = {
                _id: self.model.embeddings.word_embeddings.weight.data[
                    _id
                ].clone()
                for _id in flatten_list(self.dataset.all_class_ids)
            }

    def post_step_actions(self):
        if self.exp_handler.freeze_emotion_embeddings:
            self.model.reset_word_embeddings(self._emotion_embeddings)


class DemuxTrainerForSemEval(DemuxTrainerMixin, SemEval2018Task1EcTrainer):
    """Demux trainer. For details, see `DemuxTrainerMixin`,
    `SemEval2018Task1EcTrainer`."""

    argparse_args = deepcopy(SemEval2018Task1EcTrainer.argparse_args)
    argparse_args.update(DemuxTrainerMixin.argparse_args)


class DemuxTrainerForGoEmotions(DemuxTrainerMixin, GoEmotionsTrainer):
    """Demux trainer for GoEmotions. For everything, check
    `GoEmotionsTrainer`."""

    argparse_args = deepcopy(GoEmotionsTrainer.argparse_args)
    argparse_args.update(DemuxTrainerMixin.argparse_args)


class DemuxTrainerForFrenchElectionEmotionClusters(
    DemuxTrainerMixin, FrenchElectionTrainer
):
    """Demux trainer for French Elections. For everything, check
    `DemuxTrainerMixin`, `FrenchElectionTrainer`."""

    argparse_args = deepcopy(FrenchElectionTrainer.argparse_args)
    argparse_args.update(DemuxTrainerMixin.argparse_args)

    def get_eval_true_from_batch(self, labels: torch.Tensor) -> List[List[int]]:
        return [
            [round(y) for y in ys]
            for ys in super().get_eval_true_from_batch(labels)
        ]
