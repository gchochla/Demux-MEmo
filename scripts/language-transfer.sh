#!/bin/bash

while getopts c:r:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done


python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Arabic --dev_language English Arabic --test_language Spanish \
    --train_split train --dev_split dev --test_split train dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish --dev_language English Spanish --test_language Arabic \
    --train_split train --dev_split dev --test_split train dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language Spanish Arabic --dev_language Spanish Arabic --test_language English \
    --train_split train --dev_split dev --test_split train dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps