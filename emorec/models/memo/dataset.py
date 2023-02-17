## Author: Gireesh Mahajan

from typing import List, Optional, Dict, Any, Callable, Union
from transformers import PreTrainedTokenizerBase

import torch

from copy import deepcopy

from emorec.emorec_utils.dataset import (
    GoEmotionsDataset,
    SemEval2018Task1EcDataset,
    SemEval2018Task1EcMixDataset,
    FrenchElectionEmotionClusterDataset,
)


class MemoDatasetMixin:
    """General Memo dataset class (agnostic to specific dataset).

    Attributes:
        argparse_args: dictionary with argparse arguments and parser values.
        prompt: memo classification prompt, must include '{{}}' to insert sample text and
        '{kind_of_text}' to substitue medium of text (tweet, reddit post, etc.).
        The text to substitute can be modified via the `prompt_kwargs` argument.
    """

    argparse_args = dict(
        max_length=dict(
            default=64,
            type=int,
            help="Max total tokenized length of prompt and text",
        ),
        prompt=dict(
            default="The {kind_of_text} {{}}",
            type=str,
            help="The initial portion of the prompt that denotes the kind of text and a "
            "location for the text to classify, must be a format string that includes "
            "'{kind_of_text}' and '{{}}' for the text to classify",
        ),
    )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str = "The {kind_of_text} {{}}",
        prompt_kwargs: Dict[str, Any] = {"kind_of_text": "tweet"},
        **kwargs,
    ):
        """Init.
        Substitutes the `{{}}` in the prompt with the text to classify and
        substitutes any additional format arguments if they appear in prompt_kwargs.

        Args:
            See `SemEval2018Task1EcDataset`.
            max_length: max sequence length (including prompt).
            prompt: memo classification prompt, must include '{{}}' to insert sample text and
                '{kind_of_text}' to substitue medium of text (tweet, reddit post, etc.).
            prompt_kwargs: Additional text to substitute. Default is `{"kind_of_text": "tweet"}`.
        """

        prompt_kwargs.update({"mask_token": tokenizer.mask_token})
        self.prompt = prompt.format(**prompt_kwargs)
        self.mask_token_count = self.prompt.count(tokenizer.mask_token)
        super().__init__(
            tokenizer=tokenizer,
            **kwargs,
        )

    def encode_plus(
        self, texts: List[str], max_length: int
    ) -> Dict[str, torch.Tensor]:
        """Tokenizes input texts by using the `self.prompt` as a template.
        Truncation always happens from the right side since the prompt is
        always assumed to end with the text.

        Args:
            texts: input tweets.
            max_length: max sequence length, including prompt.

        Returns:
            `tokenizer`-style inputs.
        """

        if isinstance(texts[0], list):

            print(
                "sample prompt: ",
                self.prompt.format(self.text_preprocessor(texts[0][0])),
            )

            return [
                self.tokenizer.batch_encode_plus(
                    [
                        (self.prompt.format(self.text_preprocessor(text)))
                        for text in lang_texts
                    ],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                for lang_texts in texts
            ]

        print(
            "sample prompt: ",
            self.prompt.format(self.text_preprocessor(texts[0])),
        )

        inputs = self.tokenizer.batch_encode_plus(
            [
                self.prompt.format(self.text_preprocessor(text))
                for text in texts
            ],
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        return inputs


class MemoDatasetForSemEval(MemoDatasetMixin, SemEval2018Task1EcDataset):
    """Memo dataset for SemEval 2018 Task 1 E-c. For everything,
    check either `SemEval2018Task1EcDataset` or `MemoDatasetMixin`.
    """

    argparse_args = deepcopy(SemEval2018Task1EcDataset.argparse_args)
    argparse_args.update(MemoDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        language: str,
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        model_language: Optional[str] = None,
        logging_level: Optional[int] = None,
        prompt: str = "The {kind_of_text} {{}}",
        prompt_kwargs: Dict[str, Any] = {"kind_of_text": "tweet"},
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

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs={"max_length": max_length},
            logging_level=logging_level,
            prompt=prompt,
            prompt_kwargs=prompt_kwargs,
            # to pass these to SemEval2018Task1EcDataset, else None
            language=language,
            model_language=model_language,
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
        )


class MemoMixDatasetForSemEval(MemoDatasetMixin, SemEval2018Task1EcMixDataset):

    argparse_args = deepcopy(SemEval2018Task1EcMixDataset.argparse_args)
    argparse_args.update(MemoDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        alpha: float,
        language: str,
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        model_language: Optional[str] = None,
        logging_level: Optional[int] = None,
        prompt: str = "The {kind_of_text} {{}}",
        prompt_kwargs: Dict[str, Any] = {"kind_of_text": "tweet"},
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

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            alpha=alpha,
            encode_kwargs={"max_length": max_length},
            logging_level=logging_level,
            prompt=prompt,
            prompt_kwargs=prompt_kwargs,
            # to pass these to SemEval2018Task1EcDataset, else None
            language=language,
            model_language=model_language,
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
        )


class MemoDatasetForGoEmotions(MemoDatasetMixin, GoEmotionsDataset):
    """Memo dataset for Go Emotions. For everything,
    check either `GoEmotionsDataset` or `MemoDatasetMixin`.
    """

    argparse_args = deepcopy(GoEmotionsDataset.argparse_args)
    argparse_args.update(MemoDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        emotions_filename: Optional[str] = None,
        reddit_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        logging_level: Optional[int] = None,
        prompt: str = "The {kind_of_text} {{}}",
        prompt_kwargs: Dict[str, Any] = {"kind_of_text": "tweet"},
    ):

        self.set_emotion_order(emotions_filename)

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs={"max_length": max_length},
            prompt=prompt,
            prompt_kwargs=prompt_kwargs,
            # to pass these to GoEmotionsDataset, else None
            emotions_filename=emotions_filename,
            reddit_preprocessor=reddit_preprocessor,
            demojizer=demojizer,
            logging_level=logging_level,
        )


class MemoDatasetForFrenchElectionEmotionClusters(
    MemoDatasetMixin, FrenchElectionEmotionClusterDataset
):

    """Memo dataset for French election data. For everything,
    check either `FrenchElectionEmotionClusterDataset` or `MemoDatasetMixin`.
    """

    argparse_args = deepcopy(FrenchElectionEmotionClusterDataset.argparse_args)
    argparse_args.update(MemoDatasetMixin.argparse_args)

    def __init__(
        self,
        root_dir: str,
        splits: Union[List[str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        logging_level: Optional[int] = None,
        prompt: str = "The {kind_of_text} {{}}",
        prompt_kwargs: Dict[str, Any] = {"kind_of_text": "tweet"},
    ):

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            tokenizer=tokenizer,
            encode_kwargs={"max_length": max_length},
            logging_level=logging_level,
            prompt=prompt,
            prompt_kwargs=prompt_kwargs,
            # to pass these to FrenchElectionEmotionClusterDataset, else None
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
        )
