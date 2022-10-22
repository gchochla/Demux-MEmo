#!/bin/bash

while getopts c:r:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done

python experiments/demux.py SemEval \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language English Spanish Arabic \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --tweet_model \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language English Spanish Arabic \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language English Spanish Arabic \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0.5 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --tweet_model \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language English Spanish Arabic \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0.5 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language English Spanish Arabic \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --tweet_model \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language English Spanish Arabic \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps


