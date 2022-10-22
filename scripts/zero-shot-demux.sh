while getopts c:r:d: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
        r) reps=${OPTARG};;
        d) dir=${OPTARG};;
    esac
done


for emotion in anger joy pessimism trust
do
    echo -e "\n$emotion\n"
    python experiments/demux.py SemEval --model_name cardiffnlp/twitter-xlm-roberta-base-sentiment \
        --root_dir $dir \
        --train_language English --train_split train --dev_split dev \
        --local_correlation_coef 0.2 --local_correlation_loss intra_exp_diff \
        --num_train_epochs 20 --early_stopping_patience 3 --correct_bias \
        --device cuda:$cuda --reps $reps --excluded_emotions $emotion --freeze_emotion_embeddings
done