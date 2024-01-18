import logging
import os
import json
import random
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import langcodes
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from emorec.utils import flatten_list


class EmotionDataset(Dataset, ABC):
    """Base dataset class for multilabel emotion classification.

    Attributes:
        logger: logging module.
        tokenizer: tokenizer used for text.
        labels: dataset labels.
        inputs: model inputs.
        name: something to reflect which dataset is being used.
        excluded_emotions: emotions to discard.
        argparse_args: arguments for argparse.
    """

    argparse_args = dict(
        root_dir=dict(
            required=True,
            type=str,
            help="dataset dir",
        ),
        train_split=dict(
            default="train",
            type=str,
            nargs="+",
            help="train split(s)",
        ),
        dev_split=dict(
            type=str,
            nargs="+",
            help="development split(s)",
        ),
        test_split=dict(
            type=str,
            nargs="+",
            help="test split(s)",
        ),
        excluded_emotions=dict(
            type=str,
            nargs="*",
            help="emotions to exclude from consideration",
        ),
    )

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        encode_kwargs: Dict[str, Any],
        excluded_emotions: Optional[List[str]] = None,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            filenames: [list of] filename[s] to load from,
                data are stacked if multiple filenames provided.
            tokenizer: tokenizer to use.
            excluded_emotions: emotions to discard.
            logging_level: level to log at.
            encode_kwargs: possible `self.encode_plus` arguments.
        """

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

        self.tokenizer = tokenizer

        self.root_dir = root_dir
        self.splits = splits

        if not isinstance(splits, list):
            self.splits = [splits]

        self.name = "(" + ",".join(splits) + ")"

        self.excluded_emotions = excluded_emotions or []

        texts, labels = self.load_dataset()
        self.labels = labels
        self.inputs = self.encode_plus(texts, **encode_kwargs)

    def __len__(self):
        return len(self.labels)

    @abstractmethod
    def encode_plus(
        self, texts: List[str], **encode_kwargs
    ) -> Dict[str, torch.Tensor]:
        """Encodes input texts for models in the `transformers` format."""

    @abstractmethod
    def load_dataset(self) -> Tuple[List[str], torch.Tensor]:
        """Loads text and labels (should stack if multiple splits are provided)."""

    @property
    @abstractmethod
    def emotions(self) -> List[str]:
        """Returns a list of emotions"""

    @property
    def english_emotions(self) -> List[str]:
        """Returns emotions in English."""
        return self.emotions

    @property
    def all_emotions(self) -> List[str]:
        """Returns all emotions before exclusion"""
        return self.emotions

    @abstractmethod
    def text_preprocessor(self, text: str) -> str:
        """Returns the preprocessed text"""
        return text

    def __getitem__(
        self, index
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns a single example in the `transformers` format
        and its labels."""
        return {k: v[index] for k, v in self.inputs.items()}, self.labels[index]

    def _gettext_(self, index: int) -> Tuple[str, Dict[str, float]]:
        """Returns a single example in text, and its labels in a dict."""
        inp, label = self.__getitem__(index)

        return self.tokenizer.decode(inp["input_ids"]), {
            (emo if isinstance(emo, str) else "-".join(emo)): label[i].item()
            for i, emo in enumerate(self.emotions)
        }

    def collate_fn(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns multiple examples in the `transformers` format
        and their labels."""
        data, labels = [b[0] for b in batch], [b[1] for b in batch]
        return (
            {k: torch.stack([d[k] for d in data]) for k in data[0]},
            torch.stack(labels),
        )


class SemEval2018Task1EcDataset(EmotionDataset, ABC):
    """SemEval 2018 Task 1 E-c general dataset.

    Attributes:
        See `EmotionDataset`.
        emotions_per_lang: dict of emotion lists per language
        twitter_preprocessor: twitter-specific text preprocessor.
        demojizer: transforms emojis to text.
    """

    emotions_per_lang = {
        "english": [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "love",
            "optimism",
            "pessimism",
            "sadness",
            "surprise",
            "trust",
        ],
        "spanish": [
            "ira",
            "anticipación",
            "asco",
            "miedo",
            "alegría",
            "amor",
            "optimismo",
            "pesimismo",
            "tristeza",
            "sorpresa",
            "confianza",
        ],
        "french": [
            "colère",
            "anticipation",
            "dégoût",
            "peur",
            "joie",
            "amour",
            "optimisme",
            "pessimisme",
            "tristesse",
            "surprise",
            "confiance",
        ],
        "arabic": [
            "غضب",
            "توقع",
            "قر",
            "خوف",
            "سعادة",
            "حب",
            "تف",
            "الياس",
            "حزن",
            "اند",
            "ثقة",
        ],
    }

    argparse_args = deepcopy(EmotionDataset.argparse_args)
    argparse_args.update(
        dict(
            train_language=dict(
                required=True,
                type=str,
                nargs="+",
                help="train dataset language",
            ),
            dev_language=dict(
                type=str,
                nargs="+",
                help="dev dataset language",
            ),
            test_language=dict(
                type=str,
                nargs="+",
                help="test set language",
            ),
            model_language=dict(
                type=str,
                help="model's language",
            ),
            tweet_model=dict(
                action="store_true",
                help="whether to use model pre-trained on Twitter",
            ),
        )
    )

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        language: Union[str, List[str]],
        encode_kwargs: Dict[str, Any],
        model_language: Optional[str] = None,
        excluded_emotions: Optional[List[str]] = None,
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            language: language of dataset.
            twitter_preprocessor: twitter specific text preprocessor.
                Identity if not provided.
            demojizer: emoji handler, to be used before preprocessor.
                Identity if not provided.
            See `EmotionDataset`.
        """

        self.language = language
        if not isinstance(language, list):
            self.language = [language]

        if not model_language:
            if len(self.language) == 1:
                model_language = self.language[0]
                if model_language.split("-")[-1] == "Tr":
                    model_language = "-".join(model_language.split("-")[:-1])
            else:
                model_language = "Multimodal"

        self.model_language = model_language

        self.twitter_preprocessor = twitter_preprocessor or (lambda x: x)
        self.demojizer = demojizer or (lambda x: x)

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs=encode_kwargs,
            excluded_emotions=excluded_emotions,
            logging_level=logging_level,
        )

        self.name = "+".join(self.language) + "(" + ",".join(self.splits) + ")"
        self.emo_idx = [
            emo in self.emotion_order for emo in self.all_emotion_order
        ]

    @property
    def emotions(self) -> List[str]:
        """Emotion list for all SemEval languages."""

        emotions = self.emotions_per_lang.get(
            self.model_language.lower(), self.emotions_per_lang["english"]
        )
        emotions = [
            emo for emo in emotions if emo not in self.excluded_emotions
        ]
        return emotions

    @property
    def english_emotions(self) -> List[str]:
        english_emotions = self.emotions_per_lang["english"]
        lang_emotions = self.emotions_per_lang.get(
            self.model_language.lower(), self.emotions_per_lang["english"]
        )
        english_emotions = [
            emo_en
            for emo_en, emo in zip(english_emotions, lang_emotions)
            if emo not in self.excluded_emotions
        ]
        return english_emotions

    @property
    def emotion_order(self) -> List[str]:
        emotion_order = self.emotions_per_lang["english"]
        lang_emotions = self.emotions_per_lang.get(
            self.model_language.lower(), self.emotions_per_lang["english"]
        )
        emotion_order = [
            emo_en
            for emo_en, emo in zip(emotion_order, lang_emotions)
            if emo not in self.excluded_emotions
        ]
        return emotion_order

    @property
    def all_emotions(self) -> List[str]:
        return self.emotions_per_lang.get(
            self.model_language.lower(), self.emotions_per_lang["english"]
        )

    @property
    def all_emotion_order(self) -> List[str]:
        return self.emotions_per_lang["english"]

    def load_dataset(self) -> Tuple[List[str], torch.Tensor]:
        """Loads text and labels from dataset.

        Dataset files assumed to be delimited by `\\t`. Text
        is fetched from the 'Tweets' column (should be 2nd column),
        while the labels come from the 3rd column onward.

        Args:
            filenames: list of absolute paths.

        Returns:
            A list of text samples and their corresponding labels in a tensor.
        """

        texts = []
        labels = []
        langs = []

        lang_codes = [
            langcodes.find(lang).language.title() for lang in self.language
        ]

        split_mapping = {"train": "train", "dev": "dev", "test": "test-gold"}

        filenames = [
            [
                os.path.join(
                    self.root_dir,
                    lang.title(),
                    "E-c",
                    f"2018-E-c-{lang_code}-{split_mapping[split]}.txt",
                )
                for split in self.splits
            ]
            for lang, lang_code in zip(self.language, lang_codes)
        ]

        for lang_filenames, lang in zip(filenames, self.language):
            df = pd.concat([pd.read_csv(fn, sep="\t") for fn in lang_filenames])

            # get order of labels in the files
            file_emotion_order = list(df.columns[2:])

            texts.append(df.Tweet.values.tolist())

            # reorder labels based on our order + exclude emotions
            emotion_inds = [
                file_emotion_order.index(emotion)
                for emotion in self.all_emotion_order
            ]
            labels.append(
                torch.tensor(df.iloc[:, 2:].values[:, emotion_inds]).float()
            )

            langs.extend([lang.lower()] * len(labels[-1]))

        self.example_langs = flatten_list(langs)
        self.monolingual = len(set(self.example_langs)) == 1

        return flatten_list(texts), torch.cat(labels)

    def text_preprocessor(self, text: str) -> str:
        """Twitter-specific preprocessor."""
        return self.twitter_preprocessor(self.demojizer(text))

    def __getitem__(
        self, index
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns a single example in the `transformers` format
        and its labels."""
        if self.monolingual:
            return (
                {k: v[index] for k, v in self.inputs.items()},
                self.labels[index][self.emo_idx],
            )

        return (
            {k: v[index] for k, v in self.inputs.items()},
            self.labels[index][self.emo_idx],
            self.example_langs[index],
        )

    def collate_fn(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns multiple examples in the `transformers` format
        and their labels."""
        if self.monolingual:
            return super().collate_fn(batch)

        data, labels, example_langs = (
            [b[0] for b in batch],
            [b[1] for b in batch],
            [b[2] for b in batch],
        )
        return (
            {k: torch.stack([d[k] for d in data]) for k in data[0]},
            torch.stack(labels),
            flatten_list(example_langs),
        )


class SemEval2018Task1EcMixDataset(SemEval2018Task1EcDataset, IterableDataset):
    """SemEval 2018 Task 1 E-c Mix of languages dataset.

    Attributes:
        See `SemEval2018TaskEcDataset`.
        alpha: coefficient to normalize frequencies.
    """

    argparse_args = deepcopy(SemEval2018Task1EcDataset.argparse_args)
    argparse_args.update(
        dict(alpha=dict(type=float, help="prob. normalization exponent"))
    )

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        alpha: float,
        language: List[str],
        encode_kwargs: Dict[str, Any],
        model_language: Optional[str] = None,
        excluded_emotions: Optional[List[str]] = None,
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            alpha: coefficient to normalize frequencies.
            See `SemEval2018Task1EcDataset`.
        """

        self.language = language if isinstance(language, list) else language

        if not model_language:
            if len(self.language) == 1:
                model_language = self.language[0]
                if model_language.split("-")[-1] == "Tr":
                    model_language = "-".join(model_language.split("-")[:-1])
            else:
                model_language = "Multimodal"

        self.alpha = alpha

        self.model_language = model_language

        self.twitter_preprocessor = twitter_preprocessor or (lambda x: x)
        self.demojizer = demojizer or (lambda x: x)

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            language=language,
            model_language=model_language,
            excluded_emotions=excluded_emotions,
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
            encode_kwargs=encode_kwargs,
            logging_level=logging_level,
        )

        self.name = "+".join(self.language) + "(" + ",".join(splits) + ")"

        freqs = [len(inputs[next(iter(inputs))]) for inputs in self.inputs]
        total_freqs = sum(freqs)
        probs = [freq / total_freqs for freq in freqs]

        self.distr = torch.nn.functional.softmax(
            torch.tensor([p**alpha for p in probs]), dim=0
        ).tolist()

        cum_distr = []

        s = 0
        for p in self.distr:
            cum_distr.append(s)
            s += p

        self.cum_distr = cum_distr

        cum_lens = []
        s = 0
        for f in freqs:
            s += f
            cum_lens.append(s)
        self.cum_lens = cum_lens

    def load_dataset(self) -> Tuple[List[str], torch.Tensor]:
        """Loads text and labels from dataset.

        Dataset files assumed to be delimited by `\\t`. Text
        is fetched from the 'Tweets' column (should be 2nd column),
        while the labels come from the 3rd column onward.

        Args:
            filenames: list of absolute paths.

        Returns:
            A list of text samples and their corresponding labels in a tensor.
        """

        texts = []
        labels = []

        lang_codes = [
            langcodes.find(lang).language.title() for lang in self.language
        ]

        split_mapping = {"train": "train", "dev": "dev", "test": "test-gold"}

        filenames = [
            [
                os.path.join(
                    self.root_dir,
                    lang.title(),
                    "E-c",
                    f"2018-E-c-{lang_code}-{split_mapping[split]}.txt",
                )
                for split in self.splits
            ]
            for lang, lang_code in zip(self.language, lang_codes)
        ]

        for lang_filenames in filenames:
            df = pd.concat([pd.read_csv(fn, sep="\t") for fn in lang_filenames])

            # get order of labels in the files
            file_emotion_order = list(df.columns[2:])

            texts.append(df.Tweet.values.tolist())

            # reorder labels based on our order
            emotion_inds = [
                file_emotion_order.index(emotion)
                for emotion in self.emotion_order
            ]
            labels.append(
                torch.tensor(df.iloc[:, 2:].values[:, emotion_inds]).float()
            )

        return texts, labels

    def __iter__(self):
        while True:
            p = random.random()
            ds_idx = (
                len(self.inputs)
                - 1
                - list(reversed([p > cp for cp in self.cum_distr])).index(True)
            )
            inputs = self.inputs[ds_idx]
            labels = self.labels[ds_idx]
            idx = random.randint(0, len(inputs[next(iter(inputs))]) - 1)
            yield {k: v[idx] for k, v in inputs.items()}, labels[idx]

    def __len__(self):
        return self.cum_lens[-1]

    def __getitem__(
        self, index
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns a single example in the `transformers` format
        and its labels."""

        ds_idx = [index < l for l in self.cum_lens].index(True)

        return {
            k: v[index] for k, v in self.inputs[ds_idx].items()
        }, self.labels[ds_idx][index]

    def collate_fn(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns multiple examples in the `transformers` format
        and their labels."""
        data, labels = [b[0] for b in batch], [b[1] for b in batch]
        return (
            {k: torch.stack([d[k] for d in data]) for k in data[0]},
            torch.stack(labels),
        )


class GoEmotionsDataset(EmotionDataset, ABC):
    """GoEmotions general dataset.

    Attributes:
        See `EmotionDataset`.
        reddit_preprocessor: preprocessor for reddit.
        demojizer: transforms emojis to text.
    """

    argparse_args = deepcopy(EmotionDataset.argparse_args)
    argparse_args.update(
        dict(
            emotions_filename=dict(
                type=str, help="path to file with emotions in dataset"
            )
        )
    )

    _emotions = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
    ]

    emotion_order = None

    @property
    def emotions(self):
        return self.emotion_order or self._emotions

    # hacky fix to RecursionError
    @property
    def english_emotions(self) -> List[str]:
        return self.emotion_order or self._emotions

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        encode_kwargs: Dict[str, Any],
        emotions_filename: Optional[str] = None,
        reddit_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            reddit_preprocessor: reddit specific text preprocessor.
                Identity if not provided.
            demojizer: emoji handler, to be used before preprocessor.
                Identity if not provided.
            See `EmotionDataset`.
        """

        self.set_emotion_order(emotions_filename)
        self.reddit_preprocessor = reddit_preprocessor or (lambda x: x)
        self.demojizer = demojizer or (lambda x: x)
        super().__init__(
            root_dir, splits, tokenizer, encode_kwargs, logging_level
        )

    def set_emotion_order(self, emotions_filename: Optional[str]):
        """Resets the emotion list if `emotions_filename` is provided.
        Last emotion assumed to be neutral and therefore discarded."""
        try:
            with open(emotions_filename) as fp:
                # get rid of neutral category
                self.emotion_order = [line.strip() for line in fp.readlines()][
                    :-1
                ]
        except TypeError:
            pass

    def text_preprocessor(self, text: str) -> str:
        """Reddit-specific preprocessor."""
        return self.reddit_preprocessor(self.demojizer(text))

    def load_dataset(self) -> Tuple[List[str], torch.Tensor]:
        """Loads text and labels from dataset `filenames`.

        Dataset files assumed to be delimited by `\\t` with no headers.
        Text is fetched from the 1st column, while the labels
        are enumerated as integers in the 2nd column (comma-separated).

        Args:
            filenames: list of absolute paths.

        Returns:
            A list of text samples and their corresponding labels in a tensor.
        """

        filenames = [
            os.path.join(self.root_dir, f"{split}.tsv") for split in self.splits
        ]

        df = pd.concat(
            [pd.read_csv(fn, sep="\t", header=None) for fn in filenames]
        )
        x_train, y_train = df.iloc[:, 0].values.tolist(), df.iloc[:, 1].values

        y_train = self._multilabel_one_hot(y_train).float()

        return x_train, y_train

    def _multilabel_one_hot(self, labels: np.ndarray) -> torch.Tensor:
        """GoEmotions-specific label transformer to multilable one-hot,
        neutral emotion is discarded (represented as 0s)."""

        n_classes = len(self.emotions)

        labels = [
            list(filter(lambda x: x < n_classes, map(int, lbl.split(","))))
            for lbl in labels
        ]
        new_labels = [
            torch.nn.functional.one_hot(
                torch.tensor(lbl, dtype=int), n_classes
            ).sum(0)
            for lbl in labels
        ]
        return torch.stack(new_labels)


class FrenchElectionEmotionClusterDataset(EmotionDataset, ABC):
    """Base dataset for INCAS emotions.

    Attributes:
        Check `EmotionDataset`.
        clusters: list of emotion clusters.
        twitter_preprocessor: preprocessor for twitter.
        demojizer: transforms emojis to text.
    """

    argparse_args = deepcopy(EmotionDataset.argparse_args)
    argparse_args.update(
        dict(
            model_language=dict(
                type=str,
                help="model's language",
            )
        )
    )

    _dev_ratio = 0.1
    _test_ratio = 0.1

    clusters = dict(
        english=[
            ["anger", "hate", "contempt", "disgust"],
            ["embarrassment", "guilt", "shame", "sadness"],
            ["admiration", "love"],
            ["optimism", "hope"],
            ["joy", "happiness"],
            ["pride"],
            ["fear", "pessimism"],
            ["sarcasm", "amusement"],
            ["positive"],
            ["negative"],
        ],
        french=[
            ["colère", "haine", "mépris", "dégoût"],
            ["embarras", "culpabilité", "honte", "tristesse"],
            ["admiration", "amour"],
            ["optimisme", "espoir"],
            ["joie", "bonheur"],
            ["fierté"],
            ["peur", "pessimisme"],
            ["sarcasme", "amusement"],
            ["positif"],
            ["négatif"],
        ],
    )

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        encode_kwargs: Dict[str, Any],
        model_language: Optional[str] = None,
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            twitter_preprocessor: twitter specific text preprocessor.
                Identity if not provided.
            demojizer: emoji handler, to be used before preprocessor.
                Identity if not provided.
            model_language: French or English|Multilingual, default English.
            See `EmotionDataset`.
        """
        self.twitter_preprocessor = twitter_preprocessor or (lambda x: x)
        self.demojizer = demojizer or (lambda x: x)
        self.model_language = model_language or "english"

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs=encode_kwargs,
            logging_level=logging_level,
        )

    @property
    def emotions(self) -> List[List[str]]:
        return self.clusters.get(self.model_language.lower(), "english")

    # hacky fix to RecursionError
    @property
    def english_emotions(self) -> List[List[str]]:
        return self.clusters["english"]

    def load_dataset(self) -> Tuple[List[str], torch.Tensor]:
        """Loads text and labels from dataset `filenames`.

        Reads from a CSV where the header names for emotions begin
        with the word "emotions" followed by an underscore and
        slash-separated constituent emotions for each cluster.

        Args:
            filenames: list of absolute paths.

        Returns:
            A list of text samples and their corresponding labels in a tensor.
        """

        def _assert_get_inds(inds: List[List[int]]):
            """Checks to see that all emotions in a cluster have been found
            in at least one and at most one header, and that all emotions
            have been found in the same header.

            Args:
                inds: a list of indices for each emotion in the cluster
                    indicating which headers the emotion was found in.

            Returns:
                The cluster's index.

            Raises:
                AssertionError if emotion is not found or found more than once,
                or if emotions not in the same header.
            """
            assert all(len(emo_inds) <= 1 for emo_inds in inds)
            assert any(len(emo_inds) > 0 for emo_inds in inds)

            prev = None
            for emo_inds in inds:
                if not emo_inds:
                    continue
                if prev is None:
                    prev = emo_inds[0]
                    continue

                assert prev == emo_inds[0]

                prev = emo_inds[0]

            return prev

        df = pd.read_csv(self.root_dir, encoding="latin-1")
        file_emotion_inds = [
            i for i, col in enumerate(df.columns) if col.startswith("emotions")
        ]
        file_emotion_order = df.columns[file_emotion_inds]

        cluster_inds = []

        for cluster in self.english_emotions:
            inds = []
            for emotion in cluster:
                inds.append([])
                for i, col in enumerate(file_emotion_order):
                    if emotion in col:
                        inds[-1].append(i)

            idx = _assert_get_inds(inds)

            cluster_inds.append(idx)

        x_train = df.Text.values.tolist()
        y_train = df.iloc[:, file_emotion_inds].values[:, cluster_inds]

        # SPLITS
        _dev_size = max(1, int(self._dev_ratio * len(x_train)))
        _test_size = max(1, int(self._test_ratio * len(x_train)))
        random.seed(42)
        eval_splits_inds = random.sample(
            range(len(x_train)), _dev_size + _test_size
        )
        train_split_inds = list(
            set(range(len(x_train))).difference(eval_splits_inds)
        )
        dev_split_inds = eval_splits_inds[:_dev_size]
        test_split_inds = eval_splits_inds[_dev_size:]

        split_inds = (
            (train_split_inds if "train" in self.splits else [])
            + (dev_split_inds if "dev" in self.splits else [])
            + (test_split_inds if "test" in self.splits else [])
        )

        texts = [x_train[i] for i in split_inds]
        labels = torch.from_numpy(y_train).float()[split_inds]
        # end SPLITS

        labels = labels.where(labels <= 1, torch.tensor(1.0))

        return texts, labels

    def text_preprocessor(self, text: str) -> str:
        """Twitter-specific preprocessor."""
        return self.twitter_preprocessor(self.demojizer(text))


class PaletzDataset(EmotionDataset, ABC):
    """Dataset with Paletz's data from Lithuanian and Polish Facebook posts.

    Posts were scraped and saved into `posts.json` file, which is simply indexed by
    post id and contains the post metadata as-is from facebook_scraper.get_posts.
    """

    argparse_args = deepcopy(EmotionDataset.argparse_args)
    argparse_args.update(
        dict(
            language=dict(
                type=str,
                nargs="+",
                help="language of dataset",
                metadata=dict(
                    name=True,
                    name_transform=(
                        lambda x: x if isinstance(x, str) else "+".join(x)
                    ),
                    name_priority=1,
                ),
            ),
            model_language=dict(
                type=str,
                help="model's language",
            ),
            round_labels=dict(
                action="store_true",
                help="whether to round labels to 0 or 1",
                metadata=dict(name=True),
            ),
        )
    )

    all_emotions = [
        "anger",
        "hate",
        "contempt",
        "disgust",
        "embarrassment",
        "love",
        "admiration",
        "attraction",
        "cuteness",
        "wonder",
        "pride",
        "sadness",
        "nostalgia",
        "empathy",
        "gratitude",
        "envy",
        "fear",
        "relief",
        "confusion",
        "surprise",
        "happiness",
        "excitement",
        "amusement",
    ]
    all_emotions_short = [
        "anger",
        "hate",
        "contempt",
        "disgust",
        "embar",
        "love",
        "admir",
        "sexy",
        "cute",
        "wonder",
        "pride",
        "sad",
        "nostal",
        "empat",
        "gratitud",
        "envy",
        "fear",
        "relief",
        "confus",
        "surpr",
        "happy",
        "excite",
        "amuse",
    ]

    @property
    def emotions(self) -> List[str]:
        emos = [
            emo
            for emo in self.all_emotions
            if emo not in self.excluded_emotions
        ]
        return emos

    @property
    def emotions_short(self) -> List[str]:
        emos = [
            emo_s
            for emo_s, emo in zip(self.all_emotions_short, self.all_emotions)
            if emo not in self.excluded_emotions
        ]
        return emos

    def __init__(
        self,
        language,
        model_language,
        round_labels,
        _dev_ratio=0.1,
        _test_ratio=0.15,
        facebook_preprocessor=None,
        demojizer=None,
        **kwargs,
    ):
        self.language = language if isinstance(language, list) else [language]
        self.model_language = (model_language or "multilingual").lower()
        self.round_labels = round_labels
        self._dev_ratio = _dev_ratio
        self._test_ratio = _test_ratio
        self.facebook_preprocessor = facebook_preprocessor or (lambda x: x)
        self.demojizer = demojizer or (lambda x: x)
        super().__init__(**kwargs)

    def load_dataset(self) -> Tuple[List[Any], List[str], torch.Tensor]:
        label_fns = {
            k: os.path.join(self.root_dir, bn)
            for k, bn in zip(
                ["lithuanian", "polish"],
                ["LT_FB_all_data_master.txt", "PL_FB_all_data_master.txt"],
            )
        }
        text_fn = os.path.join(self.root_dir, "posts.json")

        with open(text_fn, "r") as fp:
            texts_json = json.load(fp)

        df = pd.concat(
            [
                pd.read_csv(fn, sep="\t")
                for k, fn in label_fns.items()
                if k in self.language
            ]
        )
        df["id"] = df["URL"].apply(lambda x: x.split("/")[-1])
        df = df[["id"] + [emo.title() + "Cons" for emo in self.emotions_short]]
        df["text"] = df["id"].apply(
            lambda x: texts_json.get(x, {}).get("text", None)
        )
        df = df.dropna()

        ids = df["id"].values.tolist()
        texts = df["text"].values.tolist()
        labels = (
            torch.from_numpy(
                df[[emo.title() + "Cons" for emo in self.emotions_short]].values
            )
            / 100
        )
        if self.round_labels:
            labels = labels.round()

        # SPLITS
        _dev_size = max(1, int(self._dev_ratio * len(texts)))
        _test_size = max(1, int(self._test_ratio * len(texts)))
        random.seed(42)
        eval_splits_inds = random.sample(
            range(len(texts)), _dev_size + _test_size
        )
        train_split_inds = list(
            set(range(len(texts))).difference(eval_splits_inds)
        )
        dev_split_inds = eval_splits_inds[:_dev_size]
        test_split_inds = eval_splits_inds[_dev_size:]

        split_inds = (
            (train_split_inds if "train" in self.splits else [])
            + (dev_split_inds if "dev" in self.splits else [])
            + (test_split_inds if "test" in self.splits else [])
        )

        texts = [texts[i] for i in split_inds]
        labels = labels[split_inds]
        ids = [ids[i] for i in split_inds]
        # end SPLITS

        return texts, labels

    def text_preprocessor(self, text: str) -> str:
        """Facebook-specific preprocessor."""
        return self.facebook_preprocessor(self.demojizer(text))
