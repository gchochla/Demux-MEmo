#!/bin/bash

while getopts c:r:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done

##### NSP #####

# no LCA
python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language English --correct_bias --num_train_epochs 3 \
    --device cuda:$cuda --reps $reps

#################################### EXP_DIFF
# exp_diff different LCAs w/o p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language English --correct_bias --num_train_epochs 6 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --device cuda:$cuda --reps $reps

# exp_diff different LCAs w/ data-driven p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language English --correct_bias --num_train_epochs 5 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --device cuda:$cuda --reps $reps

# exp_diff different LCAs w/ prior p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language English --correct_bias --num_train_epochs 4 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language English --correct_bias --num_train_epochs 4 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --device cuda:$cuda --reps $reps

### Spanish

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language Spanish --correct_bias --num_train_epochs 10 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --train_language Spanish --correct_bias --num_train_epochs 10 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --train_language Spanish --correct_bias --num_train_epochs 10 \
    --device cuda:$cuda --reps $reps

### Arabic

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language Arabic --correct_bias --num_train_epochs 8 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language Arabic --correct_bias --num_train_epochs 8 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --global_correlation_coef 0.5 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language Arabic --correct_bias --num_train_epochs 7 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --global_correlation_coef 0.1 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language Arabic --correct_bias --num_train_epochs 8 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --global_correlation_coef 0.1 --device cuda:$cuda --reps $reps


##### MLM #####

python experiments/memo.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 4 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 6 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --prompt "emoción {mask_token} en {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval \
    --root_dir $dir --train_split train dev --test_split test \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 7 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --prompt "{mask_token} المشاعر {{}} {kind_of_text} في" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head