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
    --train_language English Spanish Arabic --dev_language English \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English --dev_language English \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language Spanish \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language Spanish --dev_language Spanish \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language Arabic \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language Arabic --dev_language Arabic \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

# Translated Spanish

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish-Tr Arabic --dev_language Spanish-Tr \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --eval_steps 250 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language Spanish-Tr --dev_language Spanish-Tr \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

# run best steps with weight decay

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language English \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --max_steps $((10 * 250)) --eval_steps 250 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English --dev_language English \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --max_steps $((8 * 200)) --eval_steps 200 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language Spanish \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --max_steps $((8 * 250)) --eval_steps 250 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language Spanish --dev_language Spanish \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --max_steps $((13 * 100)) --eval_steps 200 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish Arabic --dev_language Arabic \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --max_steps $((6 * 250)) --eval_steps 250 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language Arabic --dev_language Arabic \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --max_steps $((7 * 70)) --eval_steps 200 --correct_bias \
    --device cuda:$cuda --reps $reps

# Translated Spanish

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish-Tr Arabic --dev_language Spanish-Tr \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --max_steps $((8 * 250)) --eval_steps 250 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language Spanish-Tr --dev_language Spanish-Tr \
    --train_split train --dev_split dev --early_stopping_metric jaccard_score \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 1 --max_steps $((7 * 200)) --eval_steps 200 --correct_bias \
    --device cuda:$cuda --reps $reps