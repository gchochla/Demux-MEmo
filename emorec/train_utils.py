import os
import logging
import tempfile
import operator
import itertools
import functools
from typing import Optional, Any, Dict, Iterable, Union, List, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from transformers.training_args import TrainingArguments
from transformers.utils import add_start_docstrings
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class EarlyStopping:
    """Implements early stopping in a Pytorch fashion, i.e. an init call
    where the model (you want to save) is an argument and a step function
    to be called after each evaluation.

    Attributes:
        model: nn.Module to be saved.
        tmp_fn: TemporaryFile, where to save model (can be None).
        patience: early stopping patience.
        cnt: number of early stopping steps that metric has not improved.
        delta: difference before new metric is considered better that the
            previous best one.
        higher_better: whether a higher metric is better.
        best: best metric value so far.
        best_<metric name>: other corresponding measurements can be passed
            as extra kwargs, they are stored when the main metric is stored
            by prepending 'best_' to the name.
        logger: logging module.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patience: Optional[int],
        save_model: bool = False,
        delta: float = 0,
        higher_better: bool = False,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            model: nn.Module to be saved.
            save_model: whether to save model.
            patience: early stopping patience, if `None then no early stopping.
            delta: difference before new metric is considered better that
                the previous best one.
            higher_better: whether a higher metric is better.
        """
        self.model = model
        self.tmp_fn = (
            tempfile.NamedTemporaryFile(mode="r+", suffix=".pt")
            if save_model
            else None
        )
        self.saved = False
        self.patience = patience
        self.cnt = 0
        self.delta = delta
        self.higher_better = higher_better

        self.best = None

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

    def new_best(self, metric: float) -> bool:
        """Compares the `metric` appropriately to the current best.

        Args:
            metric: metric to compare best to.

        Returns:
            True if metric is indeed better, False otherwise.
        """
        if self.best is None:
            return True
        return (
            metric > self.best + self.delta
            if self.higher_better
            else metric < self.best - self.delta
        )

    def best_str(self) -> str:
        """Formats `best` appropriately."""
        if self.best is None:
            return "None"
        return f"{self.best:.6f}"

    def step(self, metric: Optional[float], **kwargs) -> bool:
        """Compares new metric (if it is provided) with previous best,
        saves model if so (and if `model_path` was not `None`) and
        updates count of unsuccessful steps.

        Args:
            metric: metric value based on which early stopping is used.
            kwargs: all desired metrics (including the metric passed).

        Returns:
            Whether the number of unsuccesful steps has exceeded the
            patience if patience has been set, else the signal to
            continue training (aka `False`).
        """
        if self.patience is None or metric is None:
            self._save()
            return False  # no early stopping, so user gets signal to continue

        if self.new_best(metric):
            self.logger.info(
                f"Metric improved: {self.best_str()} -> {metric:.6f}"
            )
            self._store_best(metric, **kwargs)
            self.cnt = 0
            self._save()
        else:
            self.cnt += 1
            self.logger.info(
                f"Patience counter increased to {self.cnt}/{self.patience}"
            )

        return self.cnt >= self.patience

    def _save(self):
        """Saves model and logs location."""
        if self.tmp_fn is not None:
            self.saved = True
            torch.save(self.model.state_dict(), self.tmp_fn.name)
            self.tmp_fn.seek(0)
            self.logger.info("Saved model to " + self.tmp_fn.name)

    def best_model(self) -> torch.nn.Module:
        """Loads last checkpoint (if any) and returns model."""
        if self.tmp_fn is not None and self.saved:
            state = torch.load(self.tmp_fn.name)
            self.model.load_state_dict(state)
        return self.model

    def _store_best(self, metric: float, **kwargs):
        """Saves best metric and potentially other corresponsing
        measurements in kwargs."""
        self.best = metric
        for key in kwargs:
            self.__setattr__("best_" + key, kwargs[key])

    def get_metrics(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Returns accumulated best metrics.

        Returns:
            If the class was idle, nothing. Otherwise, if metrics were
            passed with kwargs in `step`, then these with the string
            `best_` prepended in their keys, else a generic dict
            with 'metric' as key and the best metric.
        """

        if self.best is None:
            return

        metrics = {
            k: v for k, v in self.__dict__.items() if k.startswith("best_")
        }

        if not metrics:
            metrics = {"metric": self.best}

        return metrics


def prod(args: Iterable[float]) -> float:
    return functools.reduce(operator.mul, args, 1)


class PairKernel:
    """Kernel that concatenates up to `order` products of the variables
    as features.

    Attributes:
        order: up to which order of relationships to create.
    """

    def __init__(self, order: int):
        """Init.

        Args:
            order: up to which order of relationships to create.
        """

        self.order = order

    # for sklearn
    def fit(self, *args, **kwargs):
        """Dummy fit function for sklearn purposes."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the input data by appending the higher
        order relationships.

        Args:
            X: input samples, data laid out across `axis==1`.

        Returns:
            Transformed matrix.
        """

        X_order = np.copy(X)

        if X.ndim == 1:
            X = X[None, ...]
        for order in range(2, self.order + 1):
            Xord = [itertools.combinations(x, order) for x in X]
            X_order = np.concatenate(
                (X_order, np.stack([[prod(x) for x in xord] for xord in Xord])),
                axis=-1,
            )

        return X_order

    def col_transform(self, cols: List[str]) -> List[str]:
        """Given names for each variable, returns what each of the
        transformed variables corresponds to.

        Args:
            cols: names of variables (columns).
        """
        cols_order = []
        cols_order += cols
        for order in range(2, self.order + 1):
            cols_comb = list(itertools.combinations(cols, order))
            cols_order += cols_comb
        return cols_order


_EPSILON = 1e-5

_TORCH_FUNCTIONS_IN_01 = {
    "sqrt": lambda x: torch.sqrt(x) + _EPSILON,
    "identity": lambda x: x + _EPSILON,
    "square": lambda x: torch.square(x) + _EPSILON,
    "log": lambda x: torch.relu(torch.log(x + _EPSILON)),
}


def _make_function_from_str(func, decreasing):
    function = _TORCH_FUNCTIONS_IN_01[func]
    if not decreasing:
        return function
    return lambda x: 1 + int("_p1" in func) - function(x)


class MultilabelConditionalWeights:
    """Estimates the probability p(x_i|x_1, .., x_{i-1}, x_{i+1}, ..., x_n)
    for binary variables (here: binary labels) and then calculates weights
    w_{1...n} for each given vector x_{1...n} wrt how probable each of its
    values are (to be 1 or to be the values x_hat_{1...n}, depending on user
    arguments)

    This implementation used logistic regression to estimate the probabilities.
    The user can specify the order of interactions that should be considered,
    e.g. by passing `order=2`, the features used in the logistic regression
    will include the products x_i * x_j \\forall i \in [n] and j \in [n]-{i}.

    Attributes:
        is_fitted: used to tell if the estimator has been fitted. If not,
            estimates will just be an array of ones, aka as if the estimator
            was not used.
        col_names: n names of variables, if any. Used only for friendlier user
            inspection of the results.
        models: the fitted logistic regressions, one per variable.
        order: order of relationship to model.
        combiner: module to get the product "groups" of the specified order.
        func: function of probability to output.
    """

    is_fitted = False
    col_names = None
    models = None

    def __init__(
        self,
        order: Optional[int] = None,
        func: Optional[str] = None,
        decreasing: bool = True,
    ):
        """Init.

        Args:
            order: order of relationship to model.
            func: function with the prediction probability as an
                independent argument. Choices are "sqrt", "identity", "square",
                "log". Prepend a "dec_" to get a decreasing function (1 - <>)
                and append a "_p1" to add 1 (<> + 1) Default is "dec_sqrt_p1", aka
                f(x) = 2 - sqrt(x).
            decreasing: whether to make the function decreasing (1-f).
        """
        self.order = order
        self.combiner = PairKernel(order)
        self.func = _make_function_from_str(func or "identity", decreasing)

    def fit(
        self,
        labels: Union[np.ndarray, torch.Tensor, List[List[int]]],
        col_names: Optional[List[str]] = None,
    ) -> "MultilabelConditionalWeights":
        """Fits the logistic regression to the provided labels
        (binary variable examples). Sets `models` to that
        `Dict[str, LogisticRegression]` if `col_names` is provided,
        else just a `List[LogisticRegression]`. Also sets `col_names`
        if provided. Also sets `is_fitted`.

        Args:
            labels: binary variables to fit to.
            col_names: whether these variables have
                specific names attach to them.

        Returns:
            `self`.
        """

        if self.order is None:
            return self

        self.is_fitted = True

        self.col_names = col_names

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        binary_labels = labels.round()

        n_models = labels.shape[1]
        models = [
            Pipeline(
                [("kernel", self.combiner), ("logistic", LogisticRegression())]
            )
            for _ in range(n_models)
        ]

        for i, model in enumerate(models):
            labels_i = np.concatenate(
                (binary_labels[:, :i], binary_labels[:, i + 1 :]), axis=-1
            )
            model.fit(labels_i, binary_labels[:, i])

        if col_names:
            self.models = {
                col_name: model.named_steps["logistic"]
                for col_name, model in zip(col_names, models)
            }
        else:
            self.models = [model.named_steps["logistic"] for model in models]

        return self

    def get_batch_weights(
        self,
        true_labels: Union[torch.Tensor, np.ndarray, List[List[int]]],
        pred_labels: Optional[
            Union[torch.Tensor, np.ndarray, List[List[int]]]
        ] = None,
    ) -> torch.Tensor:
        """Calculates and returns weights for each variable and
        for each batch in the arrays.

        Args:
            true_labels: the actual value of the variables.
            pred_labels: the value predicted by some model. If
                not provided, we assume it is the same as
                `true_labels`.

        Returns:
            Weights for each prediction (the higher the weight, the
            less probable the prediction was). Returns ones if not
            previously fitted.
        """

        if not torch.is_tensor(true_labels):
            true_labels = torch.tensor(true_labels)

        if not self.is_fitted:
            return torch.ones_like(true_labels)

        if pred_labels is not None and not torch.is_tensor(pred_labels):
            pred_labels = torch.tensor(pred_labels)
            if pred_labels.ndim == 1:
                # in case we have squeezed a batch of 1
                pred_labels.unsqueeze_(0)

        if pred_labels is None:
            pred_labels = true_labels

        probs = []
        for example_labels, example_preds in zip(true_labels, pred_labels):
            example_probs = []
            for i, model in enumerate(self.models):
                if self.col_names is not None:
                    model = self.models[model]

                emo_conditional_labels = torch.cat(
                    (example_labels[:i], example_labels[i + 1 :])
                )[None, ...]
                emo_vars = self.combiner.transform(
                    emo_conditional_labels.cpu().numpy()
                )
                prob = model.predict_proba(emo_vars)[
                    0, model.classes_.tolist().index(example_preds[i].item())
                ]
                example_probs.append(prob)
            probs.append(example_probs)

        probs = torch.tensor(probs, device=true_labels.device)
        weights = self.func(probs)

        return weights

    def get_details(
        self,
    ) -> Union[List[np.ndarray], Dict[str, Dict[str, Union[np.ndarray, List]]]]:
        """Returns the learned parameters. If no `col_names` have been
        provided, value is just a list, otherwise a dictionary with one
        element per variable containing learned parameters and corresponding
        feature names."""

        if not self.is_fitted:
            return

        if self.col_names:
            return {
                col_name: {
                    "coefficients": model.coef_[0],
                    "vars": self.combiner.col_transform(
                        self.col_names[:i] + self.col_names[i + 1 :]
                    ),
                }
                for i, (col_name, model) in enumerate(self.models.items())
            }
        return [model.coef_ for model in self.models]


_PLUTCHIK_WHEEL_ANGLES = {
    emo: np.pi * i / 8
    for i, emo in enumerate(
        [
            "fear",
            "submission",
            "trust",
            "love",
            "joy",
            "optimism",
            "anticipation",
            "agressiveness",
            "anger",
            "contempt",
            "disgust",
            "remorse",
            "sadness",
            "disapproval",
            "surprise",
            "awe",
        ]
    )
}
_PLUTCHIK_WHEEL_ANGLES["pessimism"] = _PLUTCHIK_WHEEL_ANGLES["optimism"] + np.pi

_PLUTCHIK_WHEEL_ANGLES.update(
    {
        emo_es: np.pi * i / 8
        for i, emo_es in enumerate(
            [
                "خوف",
                "استسلام",
                "ثقة",
                "حب",
                "سعادة",
                "تف",
                "توقع",
                "عدوانية",
                "غضب",
                "ازدراء",
                "قر",
                "الندم",
                "حزن",
                "الرفض",
                "اند",
                "رهبة",
            ]
        )
    }
)

_PLUTCHIK_WHEEL_ANGLES["الياس"] = _PLUTCHIK_WHEEL_ANGLES["تف"] + np.pi

_PLUTCHIK_WHEEL_ANGLES.update(
    {
        emo_es: np.pi * i / 8
        for i, emo_es in enumerate(
            [
                "miedo",
                "sumisión",
                "confianza",
                "amor",
                "alegría",
                "optimismo",
                "anticipación",
                "agresividad",
                "ira",
                "despresio",
                "asco",
                "remordimiento",
                "tristeza",
                "desaprobación",
                "sorpresa",
                "temor",
            ]
        )
    }
)

_PLUTCHIK_WHEEL_ANGLES["pesimismo"] = (
    _PLUTCHIK_WHEEL_ANGLES["optimismo"] + np.pi
)


class Correlations:
    """Wrapper around correlation matrix.

    Attributes:
        col_names: names of variables.
        func: function of correlation to return.
        corrs: correlation tensor.
        active: whether this class is active.
    """

    # in case it's inactive
    func = None
    corrs = None
    col_names = None

    def __init__(
        self,
        batched_data: Optional[
            Union[torch.Tensor, np.ndarray, List[List[int]]]
        ] = None,
        col_names: Optional[List[str]] = None,
        func: Optional[str] = None,
        normalize: bool = True,
        active: bool = True,
    ):
        """Init.

        Args:
            batched_data: input matrix, each variable is a column.
            col_names: names of columns, aka variables.
            func: function of correlation to use,
                default is decreasing identity.
            normalize: whether to project correlations to [0, 1].
            active: whether this module is active.
        """

        self.active = active

        if active:

            self.func = func or "identity"

            if batched_data is None:
                assert col_names is not None

                self.col_names = col_names

                self.corrs = torch.tensor(
                    [
                        [
                            np.cos(
                                _PLUTCHIK_WHEEL_ANGLES[col_i]
                                - _PLUTCHIK_WHEEL_ANGLES[col_j]
                            )
                            for col_j in col_names
                        ]
                        for col_i in col_names
                    ]
                )

            else:
                if torch.is_tensor(batched_data):
                    batched_data = batched_data.cpu().numpy()
                elif not isinstance(batched_data, np.ndarray):
                    batched_data = np.array(batched_data)

                self.corrs = torch.from_numpy(
                    np.corrcoef(batched_data, rowvar=False)
                )

                self.col_names = col_names

            if normalize:
                self.corrs = self.corrs / 2 + 0.5

    def _handle_index(
        self, idx: Union[int, List[int], str, List[str]]
    ) -> Optional[List[int]]:
        """Appropriately transforms each dimensions index/ices
        to a list of ints.

        Args:
            idx: index/indices of one dimension.

        Returns:
            List of integers indices.
        """

        def rec_list_elem(l):
            if not l:
                return
            if isinstance(l, list):
                return rec_list_elem(l[0])
            return l

        if hasattr(idx, "__len__") and not idx:
            return

        if isinstance(idx, (str, int)):
            idx = [idx]

        if isinstance(rec_list_elem(idx), str):
            assert (
                self.col_names is not None
            ), "str indexing only if column names provided"
            idx = [self.col_names.index(i) for i in idx]

        return idx

    def get(
        self,
        index: Tuple[Union[int, List[int], str, List[str]]],
        decreasing: bool = False,
    ) -> Optional[float]:
        """Returns correlation between variables. If multiple indices are
        provided for both dims, then all pairs are returned (inner loop is
        first dim). If not `active`, returns nothing."""

        if not self.active:
            return

        _is, js = index
        _is, js = self._handle_index(_is), self._handle_index(js)

        if _is is None or js is None:
            return

        func = _make_function_from_str(self.func, decreasing)
        return func(
            torch.tensor(
                [[self.corrs[i, j] for i in _is] for j in js]
            ).squeeze()
        )


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class MyTrainingArguments(TrainingArguments):
    """
    Args:
        early_stopping_patience: early stopping patience, defaults to `None`
            (aka no early stopping).
        model_save: whether to save model using `torch.save`,
            defaults to `False`.
        local_correlation_coef: local correlation loss coefficient, \in [0, 1]. `None` (default)
            denotes that local correlation is not used.
        local_correlation_weighting: whether to use weights based on correlations
            for local correlation terms.
        local_correlation_weighting_func: Function of normalized correlation
            to use for local correlation terms
        local_correlation_loss: local correlation loss function to use, default is
            `"exp_diff"` (original).
        multilabel_conditional_order: Order of relationship to model with
            `MultilabelConditionalWeights`, should be [0, 1].
            `None` (Default) denotes no such loss.
        multilabel_conditional_func: Function of probability to use for
            `MultilabelConditionalWeights`.
        correct_bias: if using AdamW from `transformers`, whether to
            correct bias, default is `False`.
        discard_classifier: if loading a local checkpoint, whether (not) to
            load final classifier.
    """

    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Early stopping patience, `None` "
                "(default) denotes no early stopping"
            )
        },
    )

    model_save: bool = field(
        default=False,
        metadata={
            "help": (
                "whether to save model using `torch.save`,"
                " defaults to `False`."
            )
        },
    )

    model_load_filename: Optional[str] = field(
        default=None,
        metadata={
            "help": "filename to load local checkpoint from, default to `None`."
        },
    )

    local_correlation_coef: Optional[float] = field(
        default=None,
        metadata={
            "help": "SpanEmo's correlation-aware loss coefficient, "
            "should be [0, 1]. `None` (default) denotes no such loss."
        },
    )

    local_correlation_weighting: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to weigh the local correlation terms "
            "according to correlation of pair"
        },
    )

    local_correlation_loss: Optional[str] = field(
        default="inter_exp_diff",
        metadata={
            "help": "what local correlation loss function to use (e.g. inter_exp_diff, complete_cossim)"
        },
    )

    local_correlation_weighting_func: Optional[str] = field(
        default="identity",
        metadata={
            "help": "Function of normalized correlation to use for local correlation terms"
        },
    )

    local_correlation_priors: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use prior correlations rather than "
            "data-driven ones"
        },
    )

    global_correlation_coef: Optional[float] = field(
        default=None,
        metadata={"help": "Global correlation loss coefficient"},
    )

    global_priors: bool = field(
        default=False,
        metadata={
            "help": "whether to use prior correlations rather than "
            "data-driven ones"
        },
    )

    multilabel_conditional_order: Optional[float] = field(
        default=None,
        metadata={
            "help": "Order of relationship to model with "
            "`MultilabelConditionalWeights`, should be [0, 1]. "
            "`None` (Default) denotes no such loss."
        },
    )

    multilabel_conditional_func: Optional[str] = field(
        default="sqrt_p1",
        metadata={
            "help": "Function of probability to use for "
            "`MultilabelConditionalWeights`"
        },
    )

    correct_bias: bool = field(
        default=False,
        metadata={
            "help": "if using AdamW from `transformers`, whether to "
            "correct bias, default is `False`"
        },
    )

    discard_classifier: bool = field(
        default=False,
        metadata={
            "help": "if loading a local checkpoint, "
            "whether (not) to load final classifier"
        },
    )

    early_stopping_metric: Optional[str] = field(
        default=None, metadata={"help": "early stopping metric to use"}
    )
