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
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --device cuda:$cuda --reps $reps

# Global loss coefficient

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --global_correlation_coef 100 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --global_correlation_coef 10 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --global_correlation_coef 1 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --global_correlation_coef 0.5 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --global_correlation_coef 0.1 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --global_correlation_coef 0.5 --global_priors --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --global_correlation_coef 0.1 --global_priors --device cuda:$cuda --reps $reps

#################################### EXP_DIFF
# exp_diff different LCAs w/o p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_exp_diff \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff \
    --device cuda:$cuda --reps $reps

# exp_diff different LCAs w/ data-driven p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_exp_diff --local_correlation_weighting_func identity \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity \
    --device cuda:$cuda --reps $reps

# exp_diff different LCAs w/ prior p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --device cuda:$cuda --reps $reps

#################################### COSSIM
# cossim different embedding LCAs w/o p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_cossim \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_cossim \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_cossim \
    --device cuda:$cuda --reps $reps

# cossim different LCAs w/ data-driven p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_cossim --local_correlation_weighting_func identity \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_cossim --local_correlation_weighting_func identity \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_cossim --local_correlation_weighting_func identity \
    --device cuda:$cuda --reps $reps

# cossim different LCAs w/ prior p

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_cossim --local_correlation_weighting_func identity --local_correlation_priors \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_cossim --local_correlation_weighting_func identity --local_correlation_priors \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_cossim --local_correlation_weighting_func identity --local_correlation_priors \
    --device cuda:$cuda --reps $reps

############ LOCAL + GLOBAL

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --global_correlation_coef 0.5 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --global_correlation_coef 0.5 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --global_correlation_coef 0.5 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --global_correlation_coef 0.1 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --global_correlation_coef 0.1 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --global_correlation_coef 0.1 --device cuda:$cuda --reps $reps

### Spanish

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --global_correlation_coef 0.5 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --global_correlation_coef 0.5 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --global_correlation_coef 0.5 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --global_correlation_coef 0.1 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --global_correlation_coef 0.1 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --global_correlation_coef 0.1 --device cuda:$cuda --reps $reps

### Arabic


python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --global_correlation_coef 0.5 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --global_correlation_coef 0.5 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --global_correlation_coef 0.5 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --global_correlation_coef 0.1 \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --global_correlation_coef 0.1 --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --global_correlation_coef 0.1 --device cuda:$cuda --reps $reps

##### MLM #####

python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

# different LCAs w/o p
python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_exp_diff \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

# different LCAs w/ data-driven p
python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_exp_diff --local_correlation_weighting_func identity \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

# different LCAs w/ prior p
python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss inter_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval --model_name bert-base-uncased \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language English --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss complete_exp_diff --local_correlation_weighting_func identity --local_correlation_priors \
    --output_vocab_size 11 --prompt "emotion {mask_token} in {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

# Spanish & Arabic

python experiments/memo.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Spanish --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --prompt "emoción {mask_token} en {kind_of_text} {{}}" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head

python experiments/memo.py SemEval \
    --root_dir $dir --train_split train --dev_split dev \
    --train_language Arabic --early_stopping_patience 5 --correct_bias --num_train_epochs 20 \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --prompt "{mask_token} المشاعر {{}} {kind_of_text} في" --masked_emo_type custom_head \
    --device cuda:$cuda --reps $reps \
    --description custom_head