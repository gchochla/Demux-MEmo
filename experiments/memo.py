import os
import argparse
import logging
import functools
import typing

from transformers import AutoTokenizer

# temporary hack for mro to be correct
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from emorec.emorec_utils.emotion_synonyms import EMOTION_SYNONYMS
from emorec.models.memo import (
    Memo,
    MemoDatasetForSemEval,
    MemoMixDatasetForSemEval,
    MemoTrainerForSemEval,
    MemoDatasetForGoEmotions,
    MemoTrainerForGoEmotions,
    MemoDatasetForFrenchElectionEmotionClusters,
    MemoTrainerForFrenchElectionEmotionClusters,
)
from emorec.models.demux.utils import model_selector, demojizer_selector
from emorec.utils import (
    LOGGING_FORMAT,
    twitter_preprocessor,
    reddit_preprocessor,
)
from emorec.train_utils import MyTrainingArguments
from emorec.logging_utils import ExperimentHandler

from utils import general_argparse_args, add_arguments

DATASET = {
    "SemEval": MemoMixDatasetForSemEval,
    "GoEmotions": MemoDatasetForGoEmotions,
    "FrenchElections": MemoDatasetForFrenchElectionEmotionClusters,
}
TRAINER = {
    "SemEval": MemoTrainerForSemEval,
    "GoEmotions": MemoTrainerForGoEmotions,
    "FrenchElections": MemoTrainerForFrenchElectionEmotionClusters,
}


def parse_args():
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="task", required=True)

    for task in DATASET:
        sp_task = sp.add_parser(task)
        add_arguments(sp_task, Memo.argparse_args)
        add_arguments(sp_task, DATASET[task].argparse_args)
        add_arguments(sp_task, TRAINER[task].argparse_args)
        # print(TRAINER[task].mro())
        add_arguments(sp_task, general_argparse_args)

    return parser.parse_args()


def main():
    args = parse_args()

    task = args.task
    del args.task

    assert task == "SemEval" or args.model_name_or_path is not None

    reps = args.reps
    del args.reps
    description = args.description
    del args.description

    logging_level = getattr(logging, args.logging_level)
    logging.basicConfig(
        level=logging_level,
        filename=args.logging_file,
        format=LOGGING_FORMAT,
    )
    del args.logging_level
    del args.logging_file

    if args.model_name_or_path is None:
        model_language = args.model_language or (
            args.train_language[0]
            if len(args.train_language) == 1
            else "Multilingual"
        )
        model_name, demojizer = model_selector(model_language, args.tweet_model)
        args.model_name_or_path = model_name
    else:
        demojizer = demojizer_selector(args.model_name_or_path)

    if task != "GoEmotions":
        preprocessor = twitter_preprocessor()
    else:
        preprocessor = reddit_preprocessor()

    if getattr(preprocessor, "log", False):
        logging.info(f"Text preprocessor: {preprocessor.log}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    task_dataset_kwargs = (
        dict(twitter_preprocessor=preprocessor)
        if task == "SemEval"
        else dict(
            emotions_filename=args.emotions_filename,
            reddit_preprocessor=preprocessor,
        )
        if task == "GoEmotions"
        else dict(
            twitter_preprocessor=preprocessor,
            annotation_aggregation=args.annotation_aggregation,
        )
        if task == "Protagonist"
        else dict(twitter_preprocessor=preprocessor)
    )

    if task in ("Protagonist", "FrenchElections"):
        train_dataset = DATASET[task](
            root_dir=args.root_dir,
            splits=args.train_split,
            max_length=args.max_length,
            prompt=args.prompt,
            tokenizer=tokenizer,
            demojizer=demojizer,
            **task_dataset_kwargs,
        )

        dev_dataset = (
            DATASET[task](
                root_dir=args.root_dir,
                splits=args.dev_split,
                max_length=args.max_length,
                prompt=args.prompt,
                tokenizer=tokenizer,
                demojizer=demojizer,
                **task_dataset_kwargs,
            )
            if args.dev_split is not None
            else None
        )

        test_dataset = (
            DATASET[task](
                root_dir=args.root_dir,
                splits=args.test_split,
                max_length=args.max_length,
                prompt=args.prompt,
                tokenizer=tokenizer,
                demojizer=demojizer,
                **task_dataset_kwargs,
            )
            if args.test_split is not None
            else None
        )

    elif task == "SemEval":
        if args.alpha is not None:
            train_dataset = MemoMixDatasetForSemEval(
                root_dir=args.root_dir,
                splits=args.train_split,
                language=args.train_language,
                model_language=args.model_language,
                max_length=args.max_length,
                tokenizer=tokenizer,
                demojizer=demojizer,
                alpha=args.alpha,
                prompt=args.prompt,
                **task_dataset_kwargs,
            )
        else:
            train_dataset = MemoDatasetForSemEval(
                root_dir=args.root_dir,
                splits=args.train_split,
                language=args.train_language,
                model_language=args.model_language,
                max_length=args.max_length,
                tokenizer=tokenizer,
                demojizer=demojizer,
                prompt=args.prompt,
                **task_dataset_kwargs,
            )

        model_language = args.model_language or train_dataset.model_language

        dev_dataset = (
            MemoDatasetForSemEval(
                root_dir=args.root_dir,
                splits=args.dev_split,
                language=args.dev_language or args.train_language,
                model_language=model_language,
                max_length=args.max_length,
                tokenizer=tokenizer,
                demojizer=demojizer,
                prompt=args.prompt,
                **task_dataset_kwargs,
            )
            if args.dev_split is not None
            else None
        )

        test_dataset = (
            MemoDatasetForSemEval(
                root_dir=args.root_dir,
                splits=args.test_split,
                language=args.test_language
                or args.dev_language
                or args.train_language,
                model_language=model_language,
                max_length=args.max_length,
                tokenizer=tokenizer,
                demojizer=demojizer,
                prompt=args.prompt,
                **task_dataset_kwargs,
            )
            if args.test_split is not None
            else None
        )

    else:
        train_dataset = DATASET[task](
            root_dir=args.root_dir,
            splits=args.train_split,
            max_length=args.max_length,
            tokenizer=tokenizer,
            demojizer=demojizer,
            prompt=args.prompt,
            **task_dataset_kwargs,
        )

        dev_dataset = (
            DATASET[task](
                root_dir=args.root_dir,
                splits=args.dev_split,
                max_length=args.max_length,
                tokenizer=tokenizer,
                demojizer=demojizer,
                prompt=args.prompt,
                **task_dataset_kwargs,
            )
            if args.dev_split is not None
            else None
        )

        test_dataset = (
            DATASET[task](
                root_dir=args.root_dir,
                splits=args.test_split,
                max_length=args.max_length,
                tokenizer=tokenizer,
                demojizer=demojizer,
                prompt=args.prompt,
                **task_dataset_kwargs,
            )
            if args.test_split is not None
            else None
        )

    eval_steps = (
        args.eval_steps
        or (len(train_dataset) + args.train_batch_size - 1)
        // args.train_batch_size
    )

    training_args = MyTrainingArguments(
        output_dir=None,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        no_cuda=args.device == "cpu",
        disable_tqdm=args.disable_tqdm,
        early_stopping_patience=args.early_stopping_patience,
        model_save=args.model_save,
        model_load_filename=args.model_load_filename,
        local_correlation_coef=args.local_correlation_coef,
        local_correlation_weighting=args.local_correlation_weighting_func
        is not None,
        local_correlation_weighting_func=args.local_correlation_weighting_func,
        local_correlation_priors=args.local_correlation_priors,
        local_correlation_loss=args.local_correlation_loss,
        multilabel_conditional_order=args.multilabel_conditional_order,
        multilabel_conditional_func=args.multilabel_conditional_func,
        eval_steps=eval_steps,
        max_steps=args.max_steps,
    )

    synonym_dict: typing.Dict[str, typing.List[str]] = {
        key: val
        for key, val in EMOTION_SYNONYMS.items()
        if key in train_dataset.emotions
    }
    token_synon_dict = dict()
    for key, item in synonym_dict.items():
        token_synon_dict[key] = [
            tokens[1:-1] for tokens in tokenizer([key] + item).input_ids
        ]
    token_synon_list = [
        functools.reduce(lambda x, y: x + y, tokens)
        for _, tokens in token_synon_dict.items()
    ]

    logging.info(args)
    logging.info(f"Masked token count: {train_dataset.mask_token_count}")

    for rep in range(reps):
        if reps > 1:
            print("\n", f"Rep {rep+1}", "\n")

        experiment_handler = ExperimentHandler(
            "./experiment_logs", f"Memo{task}", description=description
        )

        args = experiment_handler.set_namespace_params(args)

        if getattr(preprocessor, "log", False):
            experiment_handler.set_param("preprocessor", preprocessor.log)

        model = Memo.from_pretrained(
            args.model_name_or_path,
            dropout_prob=args.dropout_prob,
            masked_emo_type=args.masked_emo_type,
            output_vocab_size=len(train_dataset.emotions),
            mask_token_id=tokenizer.mask_token_id,
            selection_indices=token_synon_list,
            emotion_synonym_combine_func=args.emotion_synonym_combine_func,
            masked_token_count=train_dataset.mask_token_count,
            multi_mask_token_aggregation=args.multi_mask_token_aggregation,
        )

        training_args = experiment_handler.set_namespace_params(training_args)

        # setup parents and disable param for comparison
        experiment_handler.set_parent(
            "multilabel_conditional_func", "multilabel_conditional_order"
        )
        experiment_handler.set_parent(
            "local_correlation_weighting_func", "local_correlation_weighting"
        )
        experiment_handler.set_parent(
            "local_correlation_weighting_func", "local_correlation_coef"
        )
        experiment_handler.disable_params(
            ["disable_tqdm", "device", "model_save"]
        )

        # set up filename
        extra_names = {
            "_model_name_": os.path.split(args.model_name_or_path)[-1],
            "_dataset_": train_dataset.name,
            "_load_model_": os.path.basename(args.model_load_filename)
            if args.model_load_filename is not None
            else None,
        }
        if dev_dataset:
            extra_names["_dev_dataset_"] = dev_dataset.name
        if test_dataset:
            extra_names["_test_dataset"] = test_dataset.name

        experiment_handler.set_dict_params(extra_names)
        experiment_handler.disable_params(list(extra_names))

        names_list = list(extra_names) + [
            "local_correlation_coef",
            "local_correlation_weighting_func",
            "local_correlation_loss",
            "local_correlation_priors",
        ]

        if hasattr(args, "alpha"):
            names_list.append("alpha")

        experiment_handler.name_params(names_list)

        trainer = TRAINER[task](
            model,
            train_dataset,
            experiment_handler,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            logging_level=logging_level,
        )
        trainer.train()


if __name__ == "__main__":
    main()
