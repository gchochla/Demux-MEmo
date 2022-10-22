#!/bin/bash

while getopts c:r:d:f:e: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
        d) dir=${OPTARG};;
        f) frdir=${OPTARG};;
        e) explogs=${OPTARG};;
    esac
done

python experiments/demux.py FrenchElections --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $frdir \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 20 --early_stopping_patience 5 --correct_bias \
    --device cuda:$cuda --reps $reps


# pretrain models

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish Arabic \
    --train_split train dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0 --max_steps $((250 * 13)) --correct_bias \
    --device cuda:$cuda --model_save

python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $dir \
    --train_language English Spanish Arabic \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --alpha 0 --max_steps $((250 * 8)) --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --model_save \
    --model_load_filename $explogs/DemuxGoEmotions/\(train\;dev\),None,twitter-xlm-roberta-base-sentiment,None,cossim,False,0.2,intra_exp_diff,False,None_0/model.pt

### use pretrained models

python experiments/demux.py FrenchElections --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $frdir \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 20 --early_stopping_patience 5 --correct_bias \
    --model_load_filename $explogs/DemuxSemEval/English+Spanish+Arabic\(train\;dev\),None,twitter-xlm-roberta-base-sentiment,0.0,None,cossim,False,0.2,intra_exp_diff,False,None_0/model.pt \
    --device cuda:$cuda --reps $reps

python experiments/demux.py FrenchElections --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
    --root_dir $frdir \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 20 --early_stopping_patience 5 --correct_bias \
    --model_load_filename $explogs/DemuxSemEval/English+Spanish+Arabic\(train\;dev\),None,twitter-xlm-roberta-base-sentiment,0.0,None,cossim,False,0.2,intra_exp_diff,False,None_0/model.pt \
    --device cuda:$cuda --reps $reps --aggregate_logits

### complete zero-shot on french election data

for i in $(seq 1 $reps); do
    python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
        --root_dir $dir \
        --train_language English Spanish Arabic \
        --train_split train dev \
        --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
        --alpha 0 --max_steps $((250 * 13)) --correct_bias \
        --device cuda:$cuda --model_save

    python experiments/demux.py FrenchElections --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
        --root_dir $frdir \
        --train_split train --test_split dev \
        --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
        --max_steps 0 --correct_bias \
        --model_load_filename $explogs/DemuxSemEval/English+Spanish+Arabic\(train\;dev\),None,twitter-xlm-roberta-base-sentiment,0.0,None,cossim,False,0.2,intra_exp_diff,False,None_0/model.pt \
        --device cuda:$cuda --reps $reps
    
    python experiments/demux.py FrenchElections --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
        --root_dir $frdir \
        --train_split train --test_split dev \
        --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
        --max_steps 0 --correct_bias \
        --model_load_filename $explogs/DemuxSemEval/English+Spanish+Arabic\(train\;dev\),None,twitter-xlm-roberta-base-sentiment,0.0,None,cossim,False,0.2,intra_exp_diff,False,None_0/model.pt \
        --device cuda:$cuda --reps $reps --aggregate_logits
done


#################################
# use BERTweet-Fr

python experiments/demux.py SemEval --model_name Yanzhu/bertweetfr-base \
    --root_dir $dir \
    --train_language French-Tr \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 20 --early_stopping_patience 3 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py FrenchElections --model_name Yanzhu/bertweetfr-base \
    --root_dir $frdir \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 30 --early_stopping_patience 5 --correct_bias \
    --device cuda:$cuda --reps $reps

python experiments/demux.py SemEval --model_name Yanzhu/bertweetfr-base \
    --root_dir $dir \
    --train_language French-Tr \
    --train_split train dev --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 6 --correct_bias --model_save --device cuda:$cuda

python experiments/demux.py FrenchElections --model_name Yanzhu/bertweetfr-base \
    --root_dir $frdir \
    --train_split train --dev_split dev \
    --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
    --num_train_epochs 30 --early_stopping_patience 5 --correct_bias \
    --device cuda:$cuda --reps $reps --model_load_filename $explogs/DemuxSemEval/French-Tr\(train\;dev\),None,bertweetfr-base,None,None,cossim,False,0.2,intra_exp_diff,False,None_0/model.pt

# zero-shot

for i in $(seq 1 $reps); do
    python experiments/demux.py SemEval --model_name Yanzhu/bertweetfr-base \
        --root_dir $dir \
        --train_language French-Tr \
        --train_split train dev --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
        --num_train_epochs 6 --correct_bias --model_save --device cuda:$cuda

    python experiments/demux.py FrenchElections --model_name Yanzhu/bertweetfr-base \
        --root_dir $frdir \
        --train_split train --test_split dev \
        --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
        --max_steps 0 --correct_bias \
        --model_load_filename $explogs/DemuxSemEval/French-Tr\(train\;dev\),None,bertweetfr-base,None,None,cossim,False,0.2,intra_exp_diff,False,None_0/model.pt \
        --device cuda:$cuda --reps $reps
done
