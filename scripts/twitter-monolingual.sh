#!/bin/bash

while getopts c:r:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done

python experiments/demux.py SemEval --model_name vinai/bertweet-base \
    --root_dir $dir \
    --train_language English --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 20 --early_stopping_patience 3 --correct_bias \
    --early_stopping_metric jaccard_score \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name pysentimiento/robertuito-sentiment-analysis \
    --root_dir $dir \
    --train_language Spanish --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 20 --early_stopping_patience 3 --correct_bias \
    --early_stopping_metric jaccard_score \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name aubmindlab/bert-base-arabertv02-twitter \
    --root_dir $dir \
    --train_language Arabic --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 20 --early_stopping_patience 3 --correct_bias \
    --early_stopping_metric jaccard_score \
    --device cuda:$cuda --reps $reps
