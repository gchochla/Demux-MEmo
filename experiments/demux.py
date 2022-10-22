import os
import argparse
import logging

from transformers import AutoTokenizer

from emorec.models.demux import (
    Demux,
    DemuxDatasetForSemEval,
    DemuxMixDatasetForSemEval,
    DemuxDatasetForGoEmotions,
    DemuxDatasetForFrenchElectionEmotionClusters,
    DemuxTrainerForSemEval,
    DemuxTrainerForGoEmotions,
    DemuxTrainerForFrenchElectionEmotionClusters,
)
from emorec.models.demux.utils import model_selector, demojizer_selector
from emorec.utils import (
    LOGGING_FORMAT,
    reddit_preprocessor,
    twitter_preprocessor,
)
from emorec.train_utils import MyTrainingArguments
from emorec.logging_utils import ExperimentHandler

from utils import general_argparse_args, add_arguments

DATASET = {
    "SemEval": DemuxMixDatasetForSemEval,
    "GoEmotions": DemuxDatasetForGoEmotions,
    "FrenchElections": DemuxDatasetForFrenchElectionEmotionClusters,
}
TRAINER = {
    "SemEval": DemuxTrainerForSemEval,
    "GoEmotions": DemuxTrainerForGoEmotions,
    "FrenchElections": DemuxTrainerForFrenchElectionEmotionClusters,
}


def parse_args():
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="task", required=True)

    for task in DATASET:
        sp_task = sp.add_parser(task)
        add_arguments(sp_task, Demux.argparse_args)
        add_arguments(sp_task, DATASET[task].argparse_args)
        add_arguments(sp_task, TRAINER[task].argparse_args)
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
        dict(
            twitter_preprocessor=preprocessor,
        )
        if task == "SemEval"
        else dict(
            emotions_filename=args.emotions_filename,
            reddit_preprocessor=preprocessor,
        )
        if task == "GoEmotions"
        else dict(
            twitter_preprocessor=preprocessor,
            model_language=args.model_language,
        )
        if task == "FrenchElections"
        else dict(twitter_preprocessor=preprocessor)
    )

    if task == "FrenchElections":
        train_dataset = DATASET[task](
            root_dir=args.root_dir,
            splits=args.train_split,
            max_length=args.max_length,
            tokenizer=tokenizer,
            demojizer=demojizer,
            **task_dataset_kwargs,
        )

        dev_dataset = (
            DATASET[task](
                root_dir=args.root_dir,
                splits=args.dev_split,
                max_length=args.max_length,
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
                tokenizer=tokenizer,
                demojizer=demojizer,
                **task_dataset_kwargs,
            )
            if args.test_split is not None
            else None
        )

    elif task == "SemEval":

        if args.alpha is not None:
            train_dataset = DemuxMixDatasetForSemEval(
                root_dir=args.root_dir,
                splits=args.train_split,
                language=args.train_language,
                model_language=args.model_language,
                max_length=args.max_length,
                excluded_emotions=args.excluded_emotions,
                tokenizer=tokenizer,
                demojizer=demojizer,
                alpha=args.alpha,
                **task_dataset_kwargs,
            )
        else:
            train_dataset = DemuxDatasetForSemEval(
                root_dir=args.root_dir,
                splits=args.train_split,
                language=args.train_language,
                model_language=args.model_language,
                max_length=args.max_length,
                excluded_emotions=args.excluded_emotions,
                tokenizer=tokenizer,
                demojizer=demojizer,
                **task_dataset_kwargs,
            )

        model_language = args.model_language or train_dataset.model_language

        dev_dataset = (
            DemuxDatasetForSemEval(
                root_dir=args.root_dir,
                splits=args.dev_split,
                language=args.dev_language or args.train_language,
                model_language=model_language,
                max_length=args.max_length,
                tokenizer=tokenizer,
                demojizer=demojizer,
                **task_dataset_kwargs,
            )
            if args.dev_split is not None
            else None
        )

        test_dataset = (
            DemuxDatasetForSemEval(
                root_dir=args.root_dir,
                splits=args.test_split,
                language=args.test_language
                or args.dev_language
                or args.train_language,
                model_language=model_language,
                max_length=args.max_length,
                tokenizer=tokenizer,
                demojizer=demojizer,
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
            **task_dataset_kwargs,
        )

        dev_dataset = (
            DATASET[task](
                root_dir=args.root_dir,
                splits=args.dev_split,
                max_length=args.max_length,
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
                tokenizer=tokenizer,
                demojizer=demojizer,
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
        early_stopping_metric=args.early_stopping_metric,
        model_save=args.model_save,
        model_load_filename=args.model_load_filename,
        eval_steps=eval_steps,
        correct_bias=args.correct_bias,
        discard_classifier=args.discard_classifier,
        max_steps=args.max_steps,
        local_correlation_coef=args.local_correlation_coef,
        local_correlation_weighting=args.local_correlation_weighting_func
        is not None,
        local_correlation_weighting_func=args.local_correlation_weighting_func,
        local_correlation_loss=args.local_correlation_loss,
        local_correlation_priors=args.local_correlation_priors,
        multilabel_conditional_order=args.multilabel_conditional_order,
        multilabel_conditional_func=args.multilabel_conditional_func,
        global_correlation_coef=args.global_correlation_coef,
        global_priors=args.global_priors,
    )

    logging.info(args)

    for rep in range(reps):
        if reps > 1:
            print("\n", f"Rep {rep+1}", "\n")

        experiment_handler = ExperimentHandler(
            "./experiment_logs", f"Demux{task}", description=description
        )

        args = experiment_handler.set_namespace_params(args)

        if getattr(preprocessor, "log", False):
            experiment_handler.set_param("preprocessor", preprocessor.log)

        model = Demux.from_pretrained(
            args.model_name_or_path,
            class_inds=train_dataset.class_inds,
            aggregate_logits=args.aggregate_logits,
            dropout_prob=args.dropout_prob,
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
        experiment_handler.set_parent(
            "global_priors", "global_correlation_coef"
        )
        experiment_handler.set_parent(
            "global_correlation_loss", "global_correlation_coef"
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

        if hasattr(args, "excluded_emotions") and args.excluded_emotions:
            extra_names["_zexcluded_emotions_"] = "+".join(
                args.excluded_emotions
            )

        experiment_handler.set_dict_params(extra_names)
        experiment_handler.disable_params(list(extra_names))

        names_list = list(extra_names)

        if task not in ("Wassa2017", "HashtagEmotion", "Crowdflower"):
            names_list.extend(
                [
                    "local_correlation_coef",
                    "local_correlation_weighting_func",
                    "local_correlation_loss",
                    "local_correlation_priors",
                    "global_correlation_coef",
                    "global_priors",
                    "global_correlation_loss",
                ]
            )

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
