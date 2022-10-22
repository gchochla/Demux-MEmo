import os
import yaml
import pickle
import datetime
from numbers import Number
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union, Callable, List

import matplotlib.pyplot as plt
import numpy as np

# TODO: handle the addition or removal of metrics
class ExperimentHandler:
    """Module that handles everything that's necessary to compare
    and aggregate metrics across experiments.

    Attributes:
        _directory: main experiment directory.
        _experiment_name: specific experiment being ran
            (also name of experiment subfolder).
        _description: optional information to differentiate experiment
            with others that use the same hyperparams (e.g. for internal
            code change).
        _param_dict: name and value of parameters.
        _disabled_params: which params to not consider for comparison, etc.
        _name_params: which params, if any, to use to name the experiment.
        _metric_dict: name and values of metrics (across epochs/steps).
        _best_metric_dict: name and best value of metrics (based on method
            of selection designated by user or last epoch by default).
        _test_metric_dict: name and test value of metrics.
        _parent_param_dict: name of parent parameter for some parameter
            (if any). If parent param is `None` or `False`, child param's
            value is not considered.
        _dummy_active: a value of param that is considered active.
    """

    def __init__(
        self,
        experiment_root_directory: str,
        experiment_name: str,
        description: Optional[str] = None,
    ):
        """Init.

        Args:
            experiment_root_directory: main experiment directory.
            experiment_name: specific experiment being ran
                (also name of experiment subfolder).
            description: optional information to differentiate
                experiment with others that use the same hyperparams
                (e.g. for internal code change).
        """

        self._directory = experiment_root_directory
        self._experiment_name = experiment_name
        self._description = description if description is not None else ""
        self._param_dict = {}
        self._disabled_params = set()
        self._name_params = set()
        self._metric_dict = {}
        self._best_metric_dict = {}
        self._test_metric_dict = {}
        self._parent_param_dict = {}

    def __getattr__(self, name):
        if name == "model_save_filename":
            config_directory = self._get_experiment_folder(
                pattern_matching=False
            )
            model_filename = os.path.join(
                config_directory,
                (self._description + "-" if self._description else "")
                + "model.pt",
            )
            return model_filename
        assert name in self._param_dict
        return self._param_dict[name]

    def __setstate__(self, d):  # for pickling
        self.__dict__ = d

    def __getstate__(self):  # for pickling
        return self.__dict__

    def _is_inactive(self, value):
        return value is None or not value

    _dummy_active = True

    def __eq__(self, __o: object) -> bool:
        """Checks if `_experiment_name` and active params are the same."""

        def comp(eh1: ExperimentHandler, eh2: ExperimentHandler) -> bool:
            """See if active params in `eh1` are equal to the
            corresponding params of `eh2."""
            for param_name, value in eh1._param_dict.items():
                if param_name not in eh1._disabled_params:
                    parent_name = eh1._parent_param_dict.get(param_name, None)
                    # True so it seems active
                    parent_value = eh1._param_dict.get(
                        parent_name, self._dummy_active
                    )

                    # dont compare values that are inactive
                    # or have inactive parent
                    if not (
                        self._is_inactive(value)
                        or self._is_inactive(parent_value)
                    ):
                        # check if params exists and equal in eh2
                        if (
                            param_name not in eh2._param_dict
                            or value != eh2._param_dict[param_name]
                        ):
                            return False
            return True

        if not isinstance(__o, ExperimentHandler):
            return False

        if self._experiment_name != __o._experiment_name:
            return False

        if not (comp(self, __o) and comp(__o, self)):
            return False

        return True

    @classmethod
    def load_existent(
        cls, experiment_subfolder, description=None
    ) -> "ExperimentHandler":
        """Instantiates handler from existing logs.

        Args:
            experiment_subfolder: specific configuration directory.
            descriptions: optional information to differentiate
                experiment with others that use the same hyperparams
                (e.g. for internal code change).

        Returns:
            Handler.
        """

        pkl_filename = os.path.join(experiment_subfolder, "obj.pkl")
        with open(pkl_filename, "rb") as fp:
            obj = pickle.load(fp)
        obj._description = description if description is not None else ""
        return obj

    def set_parent(self, child: str, parent: str):
        """Sets parent variable of `child`."""
        assert parent in self._param_dict
        assert child in self._param_dict

        self._parent_param_dict[child] = parent

    def set_param(
        self, name: str, value: Any, parent: Optional[str] = None
    ) -> Any:
        """Sets param `name` to `value` and returns `value`.
        The parent of `name` can be specified with `parent`
        (i.e. if the parent is `False` or `None`, the value
        of this parameter won't be considered).

        Args:
            name: str name of variable.
            value: any value.
            parent: str name of optional parent variable.

        Returns:
            The value of the parameter.
        """

        self._param_dict[name] = value
        if parent is not None:
            assert parent in self._param_dict
            self.set_parent(name, parent)
        return value

    def set_namespace_params(
        self,
        arg_params: SimpleNamespace,
        parent: Optional[str] = None,
    ) -> SimpleNamespace:
        """`set_param` for names and values in a namespace.
        Returns the namespace."""
        for name, value in arg_params.__dict__.items():
            self.set_param(name, value, parent)
        return arg_params

    def set_dict_params(
        self, dict_params: Dict[str, Any], parent: Optional[str] = None
    ) -> Dict[str, Any]:
        """`set_param` for names and values in a dict.
        Returns the dict."""
        for name, value in dict_params.items():
            self.set_param(name, value, parent)
        return dict_params

    def set_metric(self, name: str, value: Any, test: bool = False) -> Any:
        """Sets metric `name` to `value` and returns `value`.

        Args:
            name: str name of variable.
            value: any value.
            test: whether this is a test or dev metric
                (default is dev, aka False).

        Returns:
            The value of the metric.
        """

        # for numpy floats, etc that mess up yaml
        if isinstance(value, int):
            value = int(value)
        elif isinstance(value, Number):
            value = float(value)

        if not test:
            if name.startswith("best_"):
                name = "_" + name
            self._metric_dict.setdefault(name, []).append(value)
        else:
            self._test_metric_dict["test_" + name] = value
        return value

    def set_dict_metrics(
        self, metrics_dict: Dict[str, Any], test: bool = False
    ) -> Dict[str, Any]:
        """`set_metric` for names and values in a dict.
        Returns the dict."""
        for name, value in metrics_dict.items():
            self.set_metric(name, value, test)
        return metrics_dict

    def disable_param(self, name: str):
        """Disables parameter `name` from comparison to other configurations."""
        assert name in self._param_dict
        self._disabled_params.add(name)

    def disable_params(self, names: List[str]):
        """Disables parameters in `names` from comparison
        to other configurations."""
        for name in names:
            self.disable_param(name)

    def name_param(self, name: str):
        """Sets parameter `name` from usage in naming of experiment."""
        assert name in self._param_dict
        self._name_params.add(name)

    def name_params(self, names: List[str]):
        """Sets name parameters."""
        for name in names:
            self.name_param(name)

    def capture_metrics(self, metric_names=None) -> Callable:
        """Decorator that captures return values of functions as metrics.

        If the function returns a list (or equivalent), the `metric_names`
        must be set, in order, to get the name of the metrics. If it is a
        dict, then we assume keys are variable names.

        Args:
            metric_names: if the returned values of the function are not in
                a dict, the names of the metrics.

        Returns:
            A function that records the metrics and then returns them.
        """

        def actual_decorator(fun):
            def wrapper(*args, **kwargs):
                results = fun(*args, **kwargs)
                if not hasattr(results, "__len__"):
                    results = [results]

                if metric_names is None:
                    assert isinstance(results, dict)
                    self.set_dict_metrics(results)
                else:
                    self.set_dict_metrics(
                        {k: v for k, v in zip(metric_names, results)}
                    )
                return results

            return wrapper

        return actual_decorator

    def _get_experiment_folder(
        self, pattern_matching: bool
    ) -> Union[str, List[str]]:
        """Returns name of directory of experiment (and creates it if
        it doens't exist).

        Args:
            pattern_matching: whether to return the name of the current
                configuration or match child variables whose parents
                are deactivated.

        Returns:
            The folder name if `pattern_matching==False`, otherwise all
            equivalent folder names (based on parent variable values).
        """

        def format_string(s):
            if isinstance(s, str):
                return (
                    s.replace(os.sep, "√").replace(",", ";").replace("=", "≈")
                )
            if hasattr(s, "__len__"):
                return ";".join([format_string(ss) for ss in s])
            return str(s)

        def strict__eq__(eh1: ExperimentHandler, eh2: ExperimentHandler):
            def sorted_dict(d):
                if not isinstance(d, dict):
                    return d
                return {k: sorted_dict(d[k]) for k in sorted(d)}

            eh1_dict = sorted_dict(eh1._param_dict)
            eh2_dict = sorted_dict(eh2._param_dict)
            return eh1_dict == eh2_dict

        experiment_subfolder = os.path.join(
            self._directory, self._experiment_name
        )

        if not os.path.exists(experiment_subfolder):
            os.makedirs(experiment_subfolder)

        exact_match_subfolder = None
        config_subfolders = []
        for subexperiment_subfolder in os.listdir(experiment_subfolder):
            abs_subexperiment_subfolder = os.path.join(
                experiment_subfolder, subexperiment_subfolder
            )
            obj = ExperimentHandler.load_existent(abs_subexperiment_subfolder)

            if strict__eq__(self, obj):
                exact_match_subfolder = abs_subexperiment_subfolder

            elif self == obj:
                config_subfolders.append(abs_subexperiment_subfolder)

        if exact_match_subfolder is None:
            exact_match_subfolder = os.path.join(
                experiment_subfolder,
                ",".join(
                    [
                        format_string(self._param_dict[param])
                        for param in sorted(self._name_params)
                    ]
                )
                + "_0",
            )
            while os.path.exists(exact_match_subfolder):
                split_name = exact_match_subfolder.split("_")
                name, index = split_name[:-1], split_name[-1]
                exact_match_subfolder = (
                    "_".join(name) + "_" + str(int(index) + 1)
                )

        if not os.path.exists(exact_match_subfolder):
            os.makedirs(exact_match_subfolder)

        return (
            [exact_match_subfolder] + config_subfolders
            if pattern_matching
            else exact_match_subfolder
        )

    def _parse_experiments_from_configs(
        self, config_directories: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Parses experiments from all directories in a single dictionary.

        Args:
            config_directories: experiment directories.

        Returns:
            A single dictionary with experiment logs.
        """

        experiments = {}
        for config_directory in config_directories:
            config_filename = os.path.join(config_directory, "metrics.yml")

            assert os.path.exists(config_filename)

            with open(config_filename) as fp:
                config_experiments = yaml.safe_load(fp)

            experiments.update(
                {
                    f"experiment_{len(experiments)+i}": config_experiments[
                        f"experiment_{i}"
                    ]
                    for i in range(len(config_experiments))
                }
            )
        return experiments

    def set_best(self, method: str, **kwargs):
        """Sets `best_metric_dict` based on lists in `metric_dict`.

        Args:
            method: how to determine which step contains the
                "best" metrics. Can be `"early_stopping"`, `"last"`.
            kwargs: must contain `"step"`, `"best_step"` or `"metric"`
                and "higher_better" if `method=="early_stopping"`.

        """
        if method == "early_stopping":
            assert ("step" in kwargs or "best_step" in kwargs) or (
                "metric" in kwargs and "higher_better" in kwargs
            ), "Must provide [best_]step or used metric for early_stopping"

            if "step" in kwargs:
                step = kwargs["step"]
            elif "best_epoch" in kwargs:
                step = kwargs["best_step"]
            else:
                metric = kwargs["metric"]
                argopt = (
                    lambda x: np.argmax(x)
                    if kwargs["higher_better"]
                    else lambda x: np.argmin(x)
                )

                step = argopt(self._metric_dict[metric])

        elif method == "last":
            step = -1

        for metric_name in self._metric_dict:
            self._best_metric_dict[f"best_{metric_name}"] = self._metric_dict[
                metric_name
            ][step]

    def log(self):
        """Logs all metrics and stores object."""
        config_directory = self._get_experiment_folder(pattern_matching=False)
        config_filename = os.path.join(config_directory, "metrics.yml")

        if os.path.exists(config_filename):
            with open(config_filename) as fp:
                experiments = yaml.safe_load(fp)
        else:
            experiments = {}

        if not self._best_metric_dict:
            self.set_best("last")

        experiment = {
            metric_name: metric_values
            for metric_name, metric_values in self._metric_dict.items()
        }
        experiment.update(
            {
                metric_name: metric_value
                for metric_name, metric_value in self._best_metric_dict.items()
            }
        )
        experiment.update(
            {
                metric_name: metric_value
                for metric_name, metric_value in self._test_metric_dict.items()
            }
        )
        if experiment:
            experiment.update({"description": self._description})

            experiments[f"experiment_{len(experiments)}"] = experiment

            with open(config_filename, "w") as fp:
                yaml.dump(experiments, fp)

        pkl_filename = os.path.join(config_directory, "obj.pkl")
        with open(pkl_filename, "wb") as fp:
            pickle.dump(self, fp)

        yml_filename = os.path.join(config_directory, "params.yml")
        with open(yml_filename, "w") as fp:
            yaml.dump(self._param_dict, fp)

    def aggregate_results(
        self, aggregation: Union[str, Dict[str, str]] = "mean"
    ):
        """Presents aggregated results of current configuration.

        Args:
            aggregation: how to aggregate best metrics across experiments.
                Can be an `str` (`"mean"` which includes std, or `"median`")
                or a dict whose keys are the metrics. The key `"other"` is
                reserved to be used as the default for each metric not specified
                in the the dict, and if that is not specified, it defaults to
                `"mean"`, which is also the general default.
        """

        def aggregate(method, values):
            if method == "mean":
                aggregated_value = (
                    f"{np.mean(values):.4f}+-{np.std(values):.4f}"
                )
            elif method == "median":
                aggregated_value = f"{np.median(values):.4f}"
            elif method == "outlier_mean":
                # preferably removes from best results
                n_outliers = 2 * len(values) // 10
                values = sorted(values)[
                    n_outliers // 2 : len(values) - (n_outliers + 1) // 2
                ]
                aggregated_value = (
                    f"{np.mean(values):.4f}+-{np.std(values):.4f}"
                )

            return aggregated_value

        if isinstance(aggregation, str):
            aggregation = {"other": aggregation}

        default_aggregation = aggregation.setdefault("other", "mean")

        config_directories = self._get_experiment_folder(pattern_matching=True)

        experiments = self._parse_experiments_from_configs(config_directories)

        best_metrics = {}
        test_metrics = {}

        for experiment in experiments.values():
            if experiment["description"] == self._description:
                for metric, value in experiment.items():
                    if metric.startswith("best_") and not isinstance(
                        value, list
                    ):
                        best_metrics.setdefault(metric, []).append(value)

                    elif metric.startswith("test_") and not isinstance(
                        value, list
                    ):
                        test_metrics.setdefault(metric, []).append(value)

        aggregated_metrics = {}
        for metric, values in best_metrics.items():
            method = aggregation.get(
                metric[len("best_") :], default_aggregation
            )

            aggregated_metrics[metric] = aggregate(method, values)

        for metric, values in test_metrics.items():
            method = aggregation.get(
                metric[len("test_") :], default_aggregation
            )
            aggregated_metrics[metric] = aggregate(method, values)

        results_filename = os.path.join(
            self._get_experiment_folder(pattern_matching=False),
            "aggregated_metrics.yml",
        )

        if os.path.exists(results_filename):
            with open(results_filename) as fp:
                results = yaml.safe_load(fp)
        else:
            results = {}

        results[self._description] = aggregated_metrics

        with open(results_filename, "w") as fp:
            yaml.dump(results, fp)

    def plot(
        self,
        aggregation: Union[str, Dict[str, str]] = "mean",
        groups: Optional[List[List[str]]] = None,
    ):
        """Plots progression of metrics of current configuration.

        Args:
            aggregation: how to aggregate best metrics across experiments.
                Can be an `str` (`"mean"` which includes std, or `"median`")
                or a dict whose keys are the metrics. The key `"other"` is
                reserved to be used as the default for each metric not specified
                in the the dict, and if that is not specified, it defaults to
                `"mean"`, which is also the general default.
            groups: how to group metrics in plots. The ones left unspecified will
                be plotted separately from the rest and each other. Note that we
                also retain the order specified in the group.
        """

        if isinstance(aggregation, str):
            aggregation = {"other": aggregation}

        default_aggregation = aggregation.setdefault("other", "mean")

        config_directories = self._get_experiment_folder(pattern_matching=True)

        experiments = self._parse_experiments_from_configs(config_directories)

        for key in list(experiments):
            experiment = experiments[key]
            description = experiment.pop("description")
            if self._description == description:
                for metric in list(experiment):
                    if metric.startswith("best_") or metric.startswith("test_"):
                        experiment.pop(metric)
                    if metric.startswith("_best"):
                        experiment[metric[1:]] = experiment.pop(metric)
                if not experiment:
                    # in case this has nothing left (e.g. a pure test entry)
                    experiments.pop(key)
            else:
                experiments.pop(key)

        if experiments:

            plot_directory = os.path.join(
                self._get_experiment_folder(pattern_matching=False),
                (f"{self._description}-" if self._description else "")
                + "plots",
            )

            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)

            all_metrics = sorted(list(next(iter(experiments.values()))))
            if groups is None:
                # assume each metric goes into its own plot
                groups = [[metric] for metric in all_metrics]
            else:
                # find if there are metrics not specified in `groups`
                left_out_metrics = sorted(
                    set(all_metrics).difference(
                        [metric for group in groups for metric in group]
                    )
                )

                # add them on their own plot
                groups.extend([[metric] for metric in left_out_metrics])

            for metrics in groups:
                # necessary because of potential early stopping
                max_length = max(
                    # assume all metrics have same length
                    [len(experiments[exp][metrics[0]]) for exp in experiments]
                )

                color = plt.cm.rainbow(np.linspace(0, 1, len(metrics)))

                for i, metric in enumerate(metrics):
                    method = aggregation.get(metric, default_aggregation)

                    group_exists = len(metrics) > 1

                    # get all values across experiments for each step
                    values_per_step = [
                        [
                            experiments[exp][metric][j]
                            for exp in experiments
                            if len(experiments[exp][metric]) > j
                        ]
                        for j in range(max_length)
                    ]

                    # aggregate
                    if method == "mean":
                        y = np.array(
                            [np.mean(values) for values in values_per_step]
                        )
                        e = np.array(
                            [np.std(values) for values in values_per_step]
                        )
                    elif method == "median":
                        y = np.array(
                            [np.mean(values) for values in values_per_step]
                        )
                        e = None

                    x = range(len(y))  # for fill_between basically

                    plt.plot(
                        x, y, label=metric if group_exists else None, c=color[i]
                    )
                    if e is not None:
                        plt.fill_between(
                            x, y + e, y - e, facecolor=color[i], alpha=0.2
                        )

                    # if single metric, then y label
                    if not group_exists:
                        plt.ylabel(
                            metric.title().replace("_", " "),
                            rotation=45,
                            labelpad=25,
                        )
                    # else, legend
                    else:
                        plt.legend(
                            bbox_to_anchor=(1.1, 1.05),
                            ncol=1,
                            fancybox=True,
                            shadow=True,
                        )

                # make xticks start from 1 and up to 11 if steps exceed that
                xticks = [1] + list(
                    range(0, max_length + 1, max(1, max_length // 10))
                )[1:]
                plt.xticks([x - 1 for x in xticks], xticks)
                plt.xlabel("Steps")
                plt.tight_layout()

                plot_basename = ",".join(metrics) + ".png"
                max_fname_len = os.statvfs("/")[-1]
                if len(plot_basename) > max_fname_len:
                    max_metric_len = (max_fname_len - len(".png")) // len(
                        metrics
                    )
                    plot_basename = (
                        ",".join(
                            # -1 for ","
                            [metric[: max_metric_len - 1] for metric in metrics]
                        )
                        + ".png"
                    )

                plot_filename = os.path.join(plot_directory, plot_basename)
                plt.savefig(plot_filename)
                plt.clf()
