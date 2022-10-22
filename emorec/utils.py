import argparse
import sys
import warnings
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Callable

import torch
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


LOGGING_FORMAT = "%(levelname)s-%(name)s(%(asctime)s)   %(message)s"


def extend_invert_attention_mask(
    attention_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Extends and inverts the attention mask because the latter
    is added internally to the logits of the attention, so we set
    what was originally a 1 to 0 and originally a o to a very
    negative value so that it effectively becomes 0 after the softmax.

    Args:
        attention_mask: 2D attention mask.
        dtype: resulting dtype.

    Returns:
        Attention mask that has been extended with the addition of
        2 singleton dimensions in between for broadcasting to the
        number of heads and the input sequence length and inverted
        for addition to logits.
    """
    extended_attention_mask = attention_mask.unsqueeze_(1).unsqueeze_(2)
    # assumes model has the same dtype everywhere
    extended_attention_mask = extended_attention_mask.to(dtype)
    inverted_extended_attention_mask = (1.0 - extended_attention_mask) * -1e4
    return inverted_extended_attention_mask


def set_parameter_requires_grad(
    model: torch.nn.Module, requires_grad: bool = False
):
    """Sets requires_grad for all the parameters in a model (in-place).

    Args:
        model: model to alter.
        requires_grad: whether the model requires grad.
    """
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def flatten_list(l: List, order: Optional[int] = None):
    """Flattens a list up to `order-1` times.

    Args:
        l: the list in question
        order: the depth of the current list,
            `None` if depth is the same for all elements
            (ergo can be discovered automatically)

    Returns:
        A list that has been flattened `order-1` times.
    """

    if not isinstance(l, list):
        l = list(l)

    if order is None:
        lc = deepcopy(l)
        order = 0
        while isinstance(lc, list) and lc:
            lc = lc[0]
            order += 1
    if order == 1:
        return l
    return [lll for ll in l for lll in flatten_list(ll, order - 1)]


def twitter_preprocessor(
    normalized_tags: Optional[List] = None, extra_tags: Optional[List] = None
) -> Callable:
    """Creates a Twitter specific text preprocessor.

    Args:
        normalized_tags: tags to anonymize, e.g. @userNamE -> user.
        extra_tags: other normalizations, e.g. Helloooooo -> hello.

    Returns:
        A function that accepts a string and returns the
        processed string.
    """

    normalized_tags = normalized_tags or ["url", "email", "phone", "user"]

    extra_tags = extra_tags or [
        "hashtag",
        "elongated",
        "allcaps",
        "repeated",
        "emphasis",
        "censored",
    ]

    def intersect_delimiters(l: List[str], demiliter: str) -> List[str]:
        new_l = []
        for elem in l:
            new_l.extend([elem, demiliter])
        return new_l

    def tag_handler_and_joiner(tokens: List[str]) -> str:
        new_tokens = []
        for token in tokens:
            for tag in normalized_tags:
                if token == f"<{tag}>":
                    token = tag
            for tag in set(extra_tags).difference(["hashtag"]):
                if token in (f"<{tag}>", f"</{tag}>"):
                    token = None
            if token:
                new_tokens.append(token)

        full_str = []
        end_pos = -1

        if "hashtag" in extra_tags:
            start_pos = -1
            while True:
                try:
                    start_pos = new_tokens.index("<hashtag>", start_pos + 1)
                    full_str.extend(
                        intersect_delimiters(
                            new_tokens[end_pos + 1 : start_pos], " "
                        )
                    )
                    end_pos = new_tokens.index("</hashtag>", start_pos + 1)
                    full_str.extend(
                        ["# "]
                        + intersect_delimiters(
                            new_tokens[start_pos + 1 : end_pos], "-"
                        )[:-1]
                        + [" "]
                    )
                except:
                    break

        full_str.extend(intersect_delimiters(new_tokens[end_pos + 1 :], " "))
        return "".join(full_str).strip()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # stop ekphrasis prints
        sys.stdout = open(os.devnull, "w")

        preprocessor = TextPreProcessor(
            normalize=normalized_tags,
            annotate=extra_tags,
            all_caps_tag="wrap",
            fix_text=False,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
        ).pre_process_doc

        # re-enable prints
        sys.stdout = sys.__stdout__

    fun = lambda x: tag_handler_and_joiner(preprocessor(x))
    fun.log = f"ekphrasis: {normalized_tags}, {extra_tags} | tag handler"
    return fun


def reddit_preprocessor(
    normalized_tags: Optional[List] = None, extra_tags: Optional[List] = None
) -> Callable:
    """Creates a Reddit specific text preprocessor.

    Args:
        normalized_tags: tags to anonymize, e.g. /u/userNamE -> user.
        extra_tags: other normalizations, e.g. Helloooooo -> hello.

    Returns:
        A function that accepts a string and returns the
        processed string.
    """

    def prepreprocessor(text):
        text = re.sub("\[NAME\]", "@name", text)
        text = re.sub("\[RELIGION\]", "religion", text)
        text = re.sub("/r/", "", text)
        text = re.sub("/u/[A-Za-z0-9_-]*", "@user", text)
        return text

    preprocessor = twitter_preprocessor()

    return lambda x: preprocessor(prepreprocessor(x))
