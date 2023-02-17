## Author: Gireesh Mahajan

from typing import Iterable, Any, Optional, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from emorec.emorec_utils.trainer import (
    SemEval2018Task1EcTrainer,
    GoEmotionsTrainer,
    FrenchElectionTrainer,
)


class MemoTrainerForSemEval(SemEval2018Task1EcTrainer):
    """Memo trainer for SemEval, check `SemEval2018Task1EcTrainer`."""

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


class MemoTrainerForGoEmotions(GoEmotionsTrainer):
    """Memo trainer for GoEmotions. For everything, check
    `GoEmotionsTrainer`."""

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


class MemoTrainerForFrenchElectionEmotionClusters(FrenchElectionTrainer):
    """Memo trainer for French Elections. For everything, check
    `FrenchElectionTrainer`."""

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
