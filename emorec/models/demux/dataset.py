from typing import List, Optional, Dict, Any, Callable, Union
from copy import deepcopy

import torch
from transformers import PreTrainedTokenizerBase

from emorec.emorec_utils.dataset import (
    SemEval2018Task1EcDataset,
    SemEval2018Task1EcMixDataset,
    GoEmotionsDataset,
    FrenchElectionEmotionClusterDataset,
    PaletzDataset,
)
from emorec.utils import flatten_list


class DemuxDatasetMixin:
    """General Demux dataset class (agnostic to specific dataset).

    Attributes:
        argparse_args: dictionary with argparse arguments and parser values.
        prompt: Demux classification prompt.
        class_inds: indices of tokens for each emotion.
    """

    argparse_args = dict(
        prompt_delimiter=dict(
            default=" ",
            type=str,
            help="what to use to separate the emotion words in the prompt",
        ),
        max_length=dict(
            default=64,
            type=int,
            help="Max total tokenized length of prompt and text",
        ),
    )

    def __init__(
        self,
        prompt_delimiter: str = " ",
        **kwargs,  # cooperative inheritance
    ):
        """Init.

        Args:
            See `SemEval2018Task1EcDataset`.
            max_length: max sequence length (including prompt).
            prompt_delimiter: string to use to separate emotions
                in prompt.
        """

        emotions_for_prompt = flatten_list(self.emotions)  # for clusters

        self.prompt = (
            prompt_delimiter.join(emotions_for_prompt[:-1])
            + " "
            + self.disjunction
            + " "
            + emotions_for_prompt[-1]
        )

        super().__init__(**kwargs)

        self.class_inds = self.get_class_inds()
        self.all_class_ids = self.get_class_ids(self.all_emotions)

    @property
    def disjunction(self):
        return "or"

    def get_class_ids(self, classes):
        return [
            self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(emotion_cls)
            )
            if isinstance(emotion_cls, str)
            # else cluster of emotions
            else [
                self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(emotion)
                )
                for emotion in emotion_cls
            ]
            for emotion_cls in classes
        ]

    def get_class_inds(
        self, example_input_ids: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """Get indices of tokens for each emotion.

        Args:
            example_input_ids: specify different tokenized sequence
                to grab class indices from (or use before initialization).

        Returns:
            A list of tensors, each containing all the indices
            for a specific emotion (in the order of `self.emotions`).
        """

        class_ids = self.get_class_ids(self.emotions)

        # NOTE: prompt SHOULD be the same in all examples
        if example_input_ids is None:
            if isinstance(self.inputs, list):
                example_input_ids = self.inputs[0]["input_ids"][0].tolist()
            else:
                example_input_ids = self.inputs["input_ids"][0].tolist()

        class_inds = []
        for ids in class_ids:
            inds = []
            if isinstance(ids[0], list):  # if cluster
                for emo_ids in ids:
                    emo_inds = []
                    for _id in emo_ids:
                        id_idx = example_input_ids.index(_id)
                        # in case it exists multiple times, turn identified instance
                        # to None (i.e. pop w/o changing indices because indices are
                        # what we are collecting)
                        example_input_ids[id_idx] = None
                        emo_inds.append(id_idx)
                    inds.append(torch.tensor(emo_inds, dtype=torch.long))
                class_inds.append(inds)
            else:  # if individual emotions
                for _id in ids:
                    id_idx = example_input_ids.index(_id)
                    # in case it exists multiple times, turn identified instance
                    # to None (i.e. pop w/o changing indices because indices are
                    # what we are collecting)
                    example_input_ids[id_idx] = None
                    inds.append(id_idx)

                class_inds.append(torch.tensor(inds, dtype=torch.long))

        return class_inds

    def encode_plus(
        self, texts: Union[List[str], List[List[str]]], max_length: int
    ) -> Dict[str, torch.Tensor]:
        """Tokenizes input texts by using the `self.prompt` as
        the first sequence of the input, the text as the second.
        Truncation always happens from the text because prompt
        is used to classify.

        Args:
            texts: input tweets.
            max_length: max sequence length, including prompt.

        Returns:
            `tokenizer`-style inputs.
        """

        if isinstance(texts[0], list):
            return [
                self.tokenizer.batch_encode_plus(
                    [
                        (self.prompt, self.text_preprocessor(text))
                        for text in lang_texts
                    ],
                    max_length=max_length,
                    truncation="only_second",
                    padding="max_length",
                    return_tensors="pt",
                )
                for lang_texts in texts
            ]

        return self.tokenizer.batch_encode_plus(
            [(self.prompt, self.text_preprocessor(text)) for text in texts],
            max_length=max_length,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
        )


class DemuxDatasetForSemEval(DemuxDatasetMixin, SemEval2018Task1EcDataset):
    """Demux dataset for SemEval 2018 Task 1 E-c. For everything,
    check either `SemEval2018Task1EcDataset` or `DemuxDatasetMixin`.

    Attributes:
        or_per_language: disjunctions for all languages.
    """

    or_per_language = {
        "english": "or",
        "french": "ou",
        "spanish": "o",
        "arabic": "أو",
    }

    argparse_args = deepcopy(SemEval2018Task1EcDataset.argparse_args)
    argparse_args.update(DemuxDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        language: str,
        model_language: Optional[str] = None,
        excluded_emotions: Optional[List[str]] = None,
        logging_level: Optional[int] = None,
        prompt_delimiter: str = " ",
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
    ):
        """Init.

        Args:
            See `DemuxDatasetMixin`, `SemEval2018Task1EcDataset`.
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
                model_language = "Multilingual"
        self.model_language = model_language

        self.excluded_emotions = excluded_emotions or []
        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs={"max_length": max_length},
            logging_level=logging_level,
            prompt_delimiter=prompt_delimiter,
            excluded_emotions=excluded_emotions,
            language=language,
            model_language=model_language,
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
        )

    @property
    def disjunction(self):
        """Disjunctions for all SemEval languages."""

        return self.or_per_language.get(
            self.model_language.lower(), self.or_per_language["english"]
        )


class DemuxMixDatasetForSemEval(
    DemuxDatasetMixin, SemEval2018Task1EcMixDataset
):
    or_per_language = {
        "english": "or",
        "french": "ou",
        "spanish": "o",
        "arabic": "أو",
    }

    argparse_args = deepcopy(SemEval2018Task1EcMixDataset.argparse_args)
    argparse_args.update(DemuxDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        alpha: float,
        language: str,
        model_language: Optional[str] = None,
        excluded_emotions: Optional[List[str]] = None,
        logging_level: Optional[int] = None,
        prompt_delimiter: str = " ",
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
    ):
        self.language = language
        if not isinstance(language, list):
            self.language = [language]

        if not model_language:
            if len(self.language) == 1:
                model_language = self.language[0]
                if model_language.split("-")[-1] == "Tr":
                    model_language = "-".join(model_language.split("-")[:-1])
            else:
                model_language = "Multilingual"
        self.model_language = model_language

        self.excluded_emotions = excluded_emotions or []

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            alpha=alpha,
            encode_kwargs={"max_length": max_length},
            logging_level=logging_level,
            prompt_delimiter=prompt_delimiter,
            excluded_emotions=excluded_emotions,
            language=language,
            model_language=model_language,
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
        )

    @property
    def disjunction(self):
        """Disjunctions for all SemEval languages."""

        return self.or_per_language.get(
            self.model_language.lower(), self.or_per_language["english"]
        )


class DemuxDatasetForGoEmotions(DemuxDatasetMixin, GoEmotionsDataset):
    """Demux dataset for GoEmotions. For everything, check either
    `DemuxDatasetMixin` or `GoEmotionsDataset`."""

    argparse_args = deepcopy(GoEmotionsDataset.argparse_args)
    argparse_args.update(DemuxDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        emotions_filename: Optional[str] = None,
        prompt_delimiter: str = " ",
        reddit_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        logging_level: Optional[int] = None,
    ):
        self.set_emotion_order(emotions_filename)

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs={"max_length": max_length},
            emotions_filename=emotions_filename,
            prompt_delimiter=prompt_delimiter,
            reddit_preprocessor=reddit_preprocessor,
            demojizer=demojizer,
            logging_level=logging_level,
        )


class DemuxDatasetForFrenchElectionEmotionClusters(
    DemuxDatasetMixin, FrenchElectionEmotionClusterDataset
):
    """Demux dataset for French election data. For everything,
    check either `FrenchElectionEmotionClusterDataset` or
    `DemuxDatasetMixin`."""

    argparse_args = deepcopy(FrenchElectionEmotionClusterDataset.argparse_args)
    argparse_args.update(DemuxDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        model_language: Optional[bool] = None,
        logging_level: Optional[int] = None,
        prompt_delimiter: str = " ",
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
    ):
        """Init.

        Args:
            See `DemuxDatasetMixin`, `FrenchElectionEmotionClusterDataset`.
        """

        self.model_language = model_language or "english"

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs={"max_length": max_length},
            logging_level=logging_level,
            prompt_delimiter=prompt_delimiter,
            model_language=model_language,
            # to pass these to FrenchElectionEmotionClusterDataset, else None
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
        )


class DemuxDatasetForPaletz(DemuxDatasetMixin, PaletzDataset):
    """Demux dataset for Paletz. For everything, check either
    `DemuxDatasetMixin` or `PaletzDataset`."""

    argparse_args = deepcopy(PaletzDataset.argparse_args)
    argparse_args.update(DemuxDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        language: str,
        round_labels: bool = True,
        model_language: Optional[str] = None,
        excluded_emotions: Optional[List[str]] = None,
        logging_level: Optional[int] = None,
        prompt_delimiter: str = " ",
        facebook_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
    ):
        self.excluded_emotions = excluded_emotions or []
        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs={"max_length": max_length},
            logging_level=logging_level,
            prompt_delimiter=prompt_delimiter,
            excluded_emotions=excluded_emotions,
            language=language,
            model_language=model_language,
            facebook_preprocessor=facebook_preprocessor,
            demojizer=demojizer,
            round_labels=round_labels,
        )
