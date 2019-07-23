#!/bin/bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED} $1${CLEAR}\n";
  fi
  exit 1
}

# Debug or not
BASE='python main.py'
if [ -n "$2" ]; then
    if [[ $2 == *"debug"* ]]; then
        BASE='python -m pudb main.py'
    fi
fi

case "$1" in
        1)
            eval $BASE
            ;;

        2)
            echo "Train with pretrained embeddings"
            eval $BASE \
                --task lm_train \
                --use_pretrained_embeddings \
                --prepared_data .data/naive_data_emb_300.pickle \
                --embeddings_path .data/embeddings/glove.6B.300d.txt \
                --saved_model_name naive_300d.pt
            ;;

        3)
            echo "Train with Expanded dataset"
            eval $BASE \
                --task lm_train \
                --expanded_dataset \
                --use_pretrained_embeddings \
                --prepared_data .data/naive_data_emb_300_exp.pickle \
                --embeddings_path .data/embeddings/glove.6B.300d.txt \
                --saved_model_name naive_300d_exp.pt
            ;;

        4)
            echo "Train with pretrained embedding, emotions, expanded dataset"
            eval $BASE \
                --task lm_train \
                --use_pretrained_embeddings \
                --prepared_data .data/_naive_data_emb_300.pickle \
                --embeddings_path .data/embeddings/glove.6B.300d.txt \
                --saved_model_name naive_300d_emo_exp.pt \
                --with_emotions \
                --expanded_dataset
            ;;
        5)
            echo "Train with ELMo"
            eval $BASE \
                --task lm_train \
                --use_pretrained_embeddings \
                --embedding_type elmo \
                --prepared_data .data/naive_data_elmo_exp.pickle  \
                --saved_model_name elmo.pt \
                --emb_dim 1024
            ;;
        6)
            echo "Train expanded"
            eval $BASE \
                --task lm_train \
                --expanded_dataset \
                --use_pretrained_embeddings \
                --embedding_type elmo \
                --prepared_data .data/naive_data_elmo.pickle  \
                --saved_model_name elmo_expanded.pt \
                --emb_dim 1024
            ;;
        7)
            echo "Running with default naive dataset and GPT embedding"
            eval $BASE \
                --use_pretrained_embeddings \
                --embedding_type gpt \
                --prepared_data .data/gpt.pickle  \
                --saved_model_name gpt_expanded.pt \
                --emb_dim 1024
            ;;
        8)
            echo "Running with default naive dataset and GPT embedding"
            eval $BASE \
                --use_pretrained_embeddings \
                --embedding_type gpt \
                --prepared_data .data/gpt.pickle  \
                --saved_model_name gpt_expanded.pt \
                --emb_dim 1024
            ;;
        9)
            echo "Classifying with pretrained Seq2Seq"
            eval $BASE \
                --task classification \
                --use_pretrained_embeddings \
                --prepared_data .data/naive_data_emb_300.pickle \
                --embeddings_path .data/embeddings/glove.6B.300d.txt \
                --saved_model_name naive_300d.pt \
                --emb_dim 300
            ;;
        10)
            echo "Classifying with pretrained expanded Seq2Seq"
            eval $BASE \
                --task classification \
                --use_pretrained_embeddings \
                --prepared_data .data/naive_data_emb_300.pickle \
                --embeddings_path .data/embeddings/glove.6B.300d.txt \
                --saved_model_name naive_300d.pt \
                --emb_dim 300
            ;;
        11)
            echo "Classifying with pretrained expanded Elmo"
            eval $BASE \
                --task classification \
                --use_pretrained_embeddings \
                --embedding_type elmo \
                --prepared_data .data/naive_data_elmo_exp.pickle \
                --saved_model_name elmo_expanded.pt \
                --emb_dim 1024
            ;;
        12)
            echo "Classifying with pretrained GPT2"
            eval $BASE \
                --task classification \
                --model 'gpt2' \
                --batch_size 32 \
                --num_epochs 50
            ;;
        13)
            echo "Zero-shot classification with pretrained GPT2"
            eval $BASE \
                --task 'zero_shot' \
                --model 'gpt2' \
                --batch_size 32 \
                --num_epochs 50
            ;;
        *)
            usage "You need to call $0 with an int option"
            exit 1

esac

