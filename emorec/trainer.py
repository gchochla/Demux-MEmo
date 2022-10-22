import logging
from typing import Optional, Union, Dict, List, Iterable, Any, Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

from emorec.utils import flatten_list
from emorec.train_utils import EarlyStopping
from emorec.logging_utils import ExperimentHandler


def result_str(results: Dict[str, float]):
    return ", ".join(
        [
            f"{key}={value:.4f}"
            if isinstance(value, float)
            else f"{key}={value}"
            for key, value in results.items()
        ]
    )


class BaseTrainer(ABC):
    """Trainer class for Twitter-201{5, 7} TMSC.

    Attributes:
        model: the model to train.
        dataset: train Twitter201X dataset.
        dev_dataset: dev Twitter201X dataset, if any.
        test_dataset: dev Twitter201X dataset, if any.
        do_eval: whether dev dataset has been set.
        do_test: whether test dataset has been set.
        args: training arguments (from `transformers`).
        verbose: whether to log/use tqdm.
        early_stopping_metric: which metric to use for early stopping.
        higher_better: whether `early_stopping_metric` is better when higher.
        argparse_args: dict of arg name and other kwargs for argparse.
        optimizer: optimizer (defined in `train`)
        scheduler: lr scheduler (defined in `train`)
    """

    early_stopping_metric = "eval_accuracy"
    higher_better = True

    argparse_args = dict(
        early_stopping_patience=dict(type=int, help="early stopping patience"),
        model_save=dict(action="store_true", help="whether to save model"),
        disable_tqdm=dict(
            action="store_true", help="disable tqdm progress bars"
        ),
        model_load_filename=dict(type=str, help="local checkpoint to load"),
        device=dict(default="cpu", type=str, help="which device to use"),
        lr=dict(default=2e-5, type=float, help="learning rate"),
        adam_beta1=dict(default=0.9, type=float, help="Adam's beta_1"),
        adam_beta2=dict(default=0.999, type=float, help="Adam's beta_2"),
        adam_epsilon=dict(default=1e-8, type=float, help="Adam's epsilon"),
        weight_decay=dict(
            default=0,
            type=float,
            help="weight decay to apply (if not zero) to all layers "
            "except all bias and LayerNorm weights in AdamW optimizer.",
        ),
        correct_bias=dict(action="store_true", help="correct bias in AdamW"),
        train_batch_size=dict(default=32, type=int, help="train batch size"),
        eval_batch_size=dict(default=32, type=int, help="eval batch size"),
        max_num_workers=dict(
            default=0,
            type=int,
            help="maximum number of workers for dataloaders",
        ),
        eval_steps=dict(
            type=int,
            help="per how many steps to evaluate on dev, default is epoch",
        ),
        max_steps=dict(default=-1, type=int, help="max number of steps"),
        num_train_epochs=dict(type=int, help="number of training epochs"),
        warmup_ratio=dict(
            default=0.1,
            type=float,
            help="ratio of training steps (not epochs)"
            " to warmup lr before linear decay",
        ),
        early_stopping_metric=dict(
            type=str, help="metric to use for early stopping"
        ),
    )

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        experiment_handler: ExperimentHandler,
        dev_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        logging_level: Optional[Union[int, str]] = None,
    ):
        """Init.

        Args:
            model: model to train.
            dataset: dataset to train on.
            train_args: training arguments.
            dev_dataset: dev dataset.
            test_dataset: test dataset.
            logging_level: level of severity of logger.
        """

        self.model = model
        self.dataset = dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.do_eval = dev_dataset is not None
        self.eval_dataset_names = dataset.name + (
            (" -> " + dev_dataset.name) if self.do_eval else ""
        )
        self.do_test = test_dataset is not None
        self.test_dataset_names = dataset.name + (
            (" -> " + test_dataset.name) if self.do_test else ""
        )
        self.exp_handler = experiment_handler

        self.early_stopping = EarlyStopping(
            self.model,
            self.exp_handler.early_stopping_patience,
            self.exp_handler.model_save,
            higher_better=self.higher_better,
            logging_level=logging_level,
        )

        self.verbose = not self.exp_handler.disable_tqdm

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

        if self.verbose:
            self.logger.debug(f"Trainer set up.")

        self.set_num_steps = (
            self.exp_handler.num_train_epochs is not None
            or self.exp_handler.max_steps > -1
        )

        if self.exp_handler.early_stopping_metric is not None:
            self.early_stopping_metric = self.exp_handler.early_stopping_metric

        assert self.set_num_steps or (
            self.early_stopping.patience is not None and self.do_eval
        )

    def train_init(self):
        """Used when training starts."""
        if self.exp_handler.model_load_filename is not None:
            self.model.load_state_dict(
                torch.load(self.exp_handler.model_load_filename)
            )

    def _save_best_model(self):
        """Loads best model to `model` attribute
        and saves to experiment folder."""
        self.model = self.early_stopping.best_model()
        if self.exp_handler.model_save:
            model_fn = self.exp_handler.model_save_filename
            torch.save(self.model.cpu().state_dict(), model_fn)
            self.logger.info(f"Saved model to {model_fn}")

    def train_end(self):
        """Used when training (and evaluation) ends."""
        self.exp_handler.log()
        self._save_best_model()
        self.exp_handler.aggregate_results()
        self.exp_handler.plot()

    def eval_init(self, data_loader: DataLoader):
        """Used when evaluation starts.

        Args:
            data_loader: DataLoader the model is going to be evaluated on.
        """

    def eval_end(self, data_loader: DataLoader):
        """Used when evaluation ends.

        Args:
            data_loader: DataLoader the model was evaluated on.
        """

    def batch_to_device(self, batch: Iterable[Any]) -> Iterable[Any]:
        """Get batch as returned by DataLoader to device."""
        batch = [
            elem.to(self.exp_handler.device)
            if torch.is_tensor(elem)
            else (
                {
                    k: (
                        v.to(self.exp_handler.device)
                        if torch.is_tensor(v)
                        else v
                    )
                    for k, v in elem.items()
                }
                if isinstance(elem, dict)
                else elem
            )
            for elem in batch
        ]
        return batch

    @abstractmethod
    def input_batch_kwargs(self, batch: Iterable[Any]) -> Dict[str, Any]:
        """Creates a kwargs dict from batch for the model."""

    def batch_labels(self, batch: Iterable[Any]):
        """Grabs labels from batch."""
        return batch[-1]

    def batch_ids(self, batch: Iterable[Any]):
        """Returns some identifier for the examples of the batch."""

    def get_logits_from_model(
        self,
        return_vals: Any,
        batch: Iterable[Any],
        data_loader: DataLoader,
        epoch: int = -1,
    ) -> torch.Tensor:
        """Grabs logits from model's return values. So far, extra arguments
        a "hacky" way to substitute images for computed embeddings when image
        encoder is frozen.

        Args:
            return_vals: return values from the model's forward function.
            batch: the batch the output was produced by.
            data_loader: the DataLoader the batch came from.
            epoch: the current epoch.

        Returns:
            The logits of the model.
        """
        return return_vals

    def batch_len(self, batch: Iterable[Any]) -> int:
        """Batch size."""
        return len(batch[0])

    def get_intermediate_repr_from_model(
        self, return_vals: Any, batch: Iterable[Any]
    ) -> Optional[torch.Tensor]:
        """Grabs intermediate representations of the model from its output,
        if necessary for some regularization loss.

        Args:
            return_vals: return values from the model's forward function.
            batch: the batch inputs came from.

        Returns:
            Some intermediate representation of the model for
                regularization losses, if necessary.
        """

    def calculate_cls_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        """Calculates train loss based on predicted logits and labels.

        Args:
            logits: model predictions.
            labels: ground truth labels.
            train: whether this is during training.

        Returns:
            Loss.
        """
        criterion = nn.CrossEntropyLoss()
        return criterion(logits, labels)

    def calculate_regularization_loss(
        self,
        intermediate_representations: Optional[torch.Tensor],
        logits: torch.Tensor,
        batch: Iterable[Any],
        train: bool,
    ) -> torch.Tensor:
        """Calculates regularization loss based on some intermediate
        representation and the batch information (like labels).

        Args:
            intermediate representations: some intermediate representation
                from the network.
            batch: the batch this representation came from.
            train: whether this is used during training.

        Returns:
            Regularization loss (or a dummy 0 tensor on the proper device).
        """
        return torch.tensor(0.0, device=self.exp_handler.device)

    def calculate_loss(
        self,
        logits: torch.Tensor,
        batch: Iterable[Any],
        train: bool,
        intermediate_representations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """Calculates loss based on predicted logits and labels.

        Args:
            logits: model predictions.
            labels: ground truth labels.
            train: whether this is during training.
            intermediate_representations

        Returns:
            Loss, train loss and regularization loss.
        """
        train_loss = self.calculate_cls_loss(
            logits, self.batch_labels(batch), train
        )

        regularization_loss = self.calculate_regularization_loss(
            intermediate_representations,
            logits,
            batch,
            train,
        )

        return (
            train_loss + regularization_loss,
            train_loss,
            regularization_loss,
        )

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initializes and returns optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.exp_handler.learning_rate,
            betas=[self.exp_handler.adam_beta1, self.exp_handler.adam_beta2],
            eps=self.exp_handler.adam_epsilon,
            weight_decay=self.exp_handler.weight_decay,
            correct_bias=self.exp_handler.correct_bias,
            no_deprecation_warning=True,
        )

    def init_optimizer_scheduler(
        self, num_batches: int
    ) -> Tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.ChainedScheduler
    ]:
        """Initializes and returns optimizer (based on `init_optimizer`)
        and scheduler.

        Args:
            num_batches: number of batches in an epoch.

        """
        optimizer = self.init_optimizer()

        if self.set_num_steps:

            if self.exp_handler.num_train_epochs:
                num_epochs = int(self.exp_handler.num_train_epochs)
                num_steps = int(num_batches * num_epochs)
            else:
                num_steps = self.exp_handler.max_steps
            warmup_steps = int(self.exp_handler.warmup_ratio * num_steps)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_steps,
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda x: 1
            )

        return optimizer, scheduler

    def pre_step_actions(self):
        """Actions before update step."""
        pass

    def post_step_actions(self):
        """Actions after update step."""
        pass

    def train(self):
        """Trains and, if a dev set or test set was provided, evaluates
        the model. (Optimizer is without bias correction)."""

        self.model = self.model.to(self.exp_handler.device)
        self.model.train()
        self.train_init()

        kwargs = (
            dict(shuffle=True)
            if not isinstance(self.dataset, torch.utils.data.IterableDataset)
            else dict()
        )

        data_loader = DataLoader(
            self.dataset,
            batch_size=self.exp_handler.train_batch_size,
            collate_fn=getattr(self.dataset, "collate_fn", None),
            num_workers=self.exp_handler.dataloader_num_workers,
            **kwargs,
        )
        if self.do_eval:
            dev_data_loader = DataLoader(
                self.dev_dataset,
                batch_size=self.exp_handler.eval_batch_size,
                collate_fn=getattr(self.dev_dataset, "collate_fn", None),
                num_workers=self.exp_handler.dataloader_num_workers,
            )
        if self.do_test:
            test_data_loader = DataLoader(
                self.test_dataset,
                batch_size=self.exp_handler.eval_batch_size,
                collate_fn=getattr(self.test_dataset, "collate_fn", None),
                num_workers=self.exp_handler.dataloader_num_workers,
            )

        self.optimizer, self.scheduler = self.init_optimizer_scheduler(
            len(data_loader)
        )

        num_epochs = int(self.exp_handler.num_train_epochs or 1)
        early_stop = False

        for epoch in range(num_epochs):

            if early_stop:
                # the early stopping check only breaks from inner loop
                break

            batch_itr = (
                tqdm(
                    data_loader,
                    desc=f"Training Epoch {epoch+1}",
                    dynamic_ncols=True,
                )
                if not self.exp_handler.disable_tqdm
                and not isinstance(
                    data_loader.dataset, torch.utils.data.IterableDataset
                )
                else data_loader
            )

            for step, batch in enumerate(batch_itr):

                step += epoch * len(data_loader)

                early_stop = (
                    self.exp_handler.max_steps > -1
                    and step >= self.exp_handler.max_steps
                )
                if early_stop:
                    self.logger.info("Forcibly stopping training")
                    break

                if step % self.exp_handler.eval_steps == 0:
                    # FIRST step of current collection of evaluation steps
                    # will always init when epoch == 0, step == 0
                    train_loss = 0.0
                    cum_regularization_loss = 0.0
                    n_samples = 0

                batch = self.batch_to_device(batch)

                self.pre_step_actions()

                return_vals = self.model(**self.input_batch_kwargs(batch))
                logits = self.get_logits_from_model(
                    return_vals, batch, data_loader, epoch
                )
                inter_repr = self.get_intermediate_repr_from_model(
                    return_vals, batch
                )

                loss, cls_loss, reg_loss = self.calculate_loss(
                    logits,
                    batch,
                    train=True,
                    intermediate_representations=inter_repr,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.post_step_actions()

                train_loss += cls_loss.item() * self.batch_len(batch)
                cum_regularization_loss += reg_loss.item() * self.batch_len(
                    batch
                )
                n_samples += self.batch_len(batch)

                if (step + 1) % self.exp_handler.eval_steps == 0:
                    # LAST step of current collection of evaluation steps

                    train_loss /= n_samples
                    cum_regularization_loss /= n_samples

                    results = dict(
                        train_loss=train_loss,
                        regularization_loss=cum_regularization_loss,
                    )

                    if self.do_eval:
                        results.update(
                            self.evaluate(
                                dev_data_loader,
                                f"Evaluating after {step+1} steps (epoch {epoch+1})",
                            )
                        )

                    self.exp_handler.set_dict_metrics(results)

                    self.logger.info(
                        f"Step {step+1} (epoch {epoch+1}) metrics on "
                        + self.eval_dataset_names
                        + ": "
                        + result_str(results)
                    )

                    early_stop = self.early_stopping.step(
                        results.get(self.early_stopping_metric, None),
                        **{**results, "epoch": epoch + 1, "step": step + 1},
                    )
                    if early_stop:
                        self.logger.info(
                            "Early stopping at step "
                            f"{step + 1} (epoch {epoch+1})"
                        )
                        break

        early_stopping_metrics = self.early_stopping.get_metrics()
        if early_stopping_metrics is not None:
            self.logger.info(
                f"Best metrics based on {self.early_stopping_metric} "
                f"on {self.eval_dataset_names}: "
                + result_str(early_stopping_metrics)
            )
            self.exp_handler.set_best(
                "early_stopping",
                metric=self.early_stopping_metric,
                higher_better=True,
            )

        if self.do_test:
            results = self.evaluate(test_data_loader, "Testing")
            self.logger.info(
                f"Testing metrics for {self.test_dataset_names}: "
                + result_str(results)
            )
            self.exp_handler.set_dict_metrics(results, test=True)

        self.train_end()

    def evaluate(
        self,
        data_loader: DataLoader,
        tqdm_message: Optional[str] = "Evaluation",
    ):
        """Evaluates model on `data_loader`.

        Args:
            data_loader: dataset to evaluate on.
            tqdm_message: what to print if tqdm is used.
        """

        self.model.eval()
        self.eval_init(data_loader)

        batch_itr = (
            tqdm(data_loader, desc=tqdm_message, dynamic_ncols=True)
            if not self.exp_handler.disable_tqdm
            and not isinstance(
                data_loader.dataset, torch.utils.data.IterableDataset
            )
            else data_loader
        )

        eval_preds = []
        eval_true = []
        eval_ids = []
        eval_loss = 0.0
        eval_reg_loss = 0.0

        for batch in batch_itr:
            batch = self.batch_to_device(batch)

            with torch.no_grad():
                return_vals = self.model(**self.input_batch_kwargs(batch))

            logits = self.get_logits_from_model(return_vals, batch, data_loader)
            inter_reprs = self.get_intermediate_repr_from_model(
                return_vals, batch
            )

            _, cls_loss, reg_loss = self.calculate_loss(
                logits,
                batch,
                train=False,
                intermediate_representations=inter_reprs,
            )

            eval_loss += cls_loss.item() * self.batch_len(batch)
            eval_reg_loss += reg_loss.item() * self.batch_len(batch)

            eval_preds.extend(self.get_eval_preds_from_batch(logits))
            eval_true.extend(
                self.get_eval_true_from_batch(self.batch_labels(batch))
            )
            ids = self.batch_ids(batch)
            if ids:
                eval_ids.extend(ids)

        ### compute eval metrics
        eval_loss /= len(data_loader.dataset)
        eval_reg_loss /= len(data_loader.dataset)

        results = dict(
            eval_loss=eval_loss, eval_regularization_loss=eval_reg_loss
        )
        results.update(
            self.evaluation_metrics(
                eval_true,
                eval_preds,
                data_loader=data_loader,
                eval_ids=eval_ids,
            )
        )

        self.model.train()
        self.eval_end(data_loader)

        return results

    def get_eval_preds_from_batch(
        self, logits: torch.Tensor
    ) -> List[List[int]]:
        """Returns predictions in batch based on logits.
        List of list is chosen for compatibility with multilabel
        setting (e.g. multiple targets)."""
        batch_preds = [
            # tolist yields a number for 0 dim tensor
            ex_logits.argmax(dim=-1).tolist()
            for ex_logits in logits
        ]
        # turn all to lists for consistency across single and multi target settings
        batch_preds = [
            ([pred] if isinstance(pred, int) else pred) for pred in batch_preds
        ]
        return batch_preds

    def get_eval_true_from_batch(self, labels: torch.Tensor) -> List[List[int]]:
        """Returns ground-truth. List of list is chosen for
        compatibility with multilabel setting (e.g. multiple targets)."""
        # turn all to lists for consistency across single and multi target settings
        batch_labels = [
            [label.item()] if label.ndim == 0 else label.tolist()
            for label in labels
        ]
        return batch_labels

    def evaluation_metrics(
        self,
        eval_true: List[List[int]],
        eval_preds: List[List[int]],
        data_loader: DataLoader,
        eval_ids: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """Computes evaluation metrics (beyond evaluation loss).

        Args:
            eval_true: ground-truth labels.
            eval_preds: predictions.
            data_loader: DataLoader where data came from.

        Returns:
            A dict of metrics.
        """
        _, _, macro_f1_score, _ = precision_recall_fscore_support(
            flatten_list(eval_true),
            flatten_list(eval_preds),
            average="macro",
            zero_division=0,
        )

        eval_accuracy = np.mean(
            [
                pred == label
                for preds, labels in zip(eval_preds, eval_true)
                for pred, label in zip(preds, labels)
            ]
        )

        results = dict(
            eval_accuracy=eval_accuracy,
            macro_f1_score=macro_f1_score,
        )

        return results
