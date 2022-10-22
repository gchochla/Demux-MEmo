from typing import Callable, Tuple, Optional

from emoji import demojize


def demojizer_selector(
    model_name: str, delimiters: Tuple[str] = ("(", ")")
) -> Callable:
    """Fetches demojizer based on model.

    Args:
        model_name: language model name (from `transformers`).
        delimiters: strings to delimit the emoji description by.

    Returns:
        Demojizer function (identity if correspondence not set).
    """
    demojizers = {
        "vinai/bertweet-base": lambda x: x,
        "bert-base-uncased": lambda x: demojize(
            x, language="en", delimiters=delimiters
        ).replace("_", " "),
        "Yanzhu/bertweetfr-base": lambda x: demojize(
            x, language="fr", delimiters=delimiters
        ).replace("_", " "),
        "flaubert/flaubert_base_uncased": lambda x: demojize(
            x, language="fr", delimiters=delimiters
        ).replace("_", " "),
        "dccuchile/bert-base-spanish-wwm-uncased": lambda x: demojize(
            x, language="es", delimiters=delimiters
        ).replace("_", " "),
        "asafaya/bert-base-arabic": lambda x: x,
        "cardiffnlp/twitter-xlm-roberta-base-sentiment": lambda x: x,
        "bert-base-multilingual-uncased": lambda x: demojize(
            x, language="en", delimiters=delimiters
        ).replace("_", " "),
    }
    return demojizers.get(model_name, lambda x: x)


def model_selector(
    language: str, trained_on_twitter: bool = True
) -> Tuple[str, Callable]:
    """Utility function to allow CL user to pick a model
    based on language and preference of whether the model
    has been trained on Twitter or not.

    Args:
        language: language of model ("english", "french", "spanish" or "arabic",
            anything else yields multilingual stuff).
        trained_on_twitter: whether model should be pre-trained
            on Twitter or not.

    Returns:
        The language model name (as a `transformers` model name) and
        a function to handle emojis in text.
    """

    language = language.lower()
    delimiters = ("(", ")")

    if language == "english":
        if trained_on_twitter:
            model_name = "vinai/bertweet-base"
        else:
            model_name = "bert-base-uncased"

    elif language == "french":
        if trained_on_twitter:
            model_name = "Yanzhu/bertweetfr-base"
        else:
            model_name = "flaubert/flaubert_base_uncased"

    elif language == "spanish":
        if trained_on_twitter:
            model_name = "pysentimiento/robertuito-sentiment-analysis"
        else:
            model_name = "dccuchile/bert-base-spanish-wwm-uncased"

    elif language == "arabic":
        if trained_on_twitter:
            model_name = "aubmindlab/bert-base-arabertv02-twitter"
        else:
            model_name = "asafaya/bert-base-arabic"

    else:  # multilingual
        if trained_on_twitter:
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        else:
            model_name = "bert-base-multilingual-uncased"

    return model_name, demojizer_selector(model_name, delimiters)
