#!/usr/bin/env bash

if [[ "$#" -ne 4 ]]; then
    echo "Usage is $0 [batch size] [dropout rate] [layers:ia,lstm,conv,usr_embed] [usr embed files]"
    exit
fi

BATCH_SIZE=$1
DROPOUT_RATE=$2
LAYERS=$3
USR_EMBED_FILES=$4

python -u src/main.py \
    --tweets_col tweets \
    --buckets_col train_data \
    --train_buckets 9,2,6,3,0,4,5,7 \
    --valid_buckets 8 \
    --test_buckets 1 \
    --batch_size ${BATCH_SIZE} \
    --grad_clip_norm 0.1 \
    --dropout_rate ${DROPOUT_RATE} \
    --min_word_freq 2 \
    --min_doc_len 3 \
    --max_doc_len 40 \
    --word_embeds_col embeddings_glove \
    --word_embed_dim 100 \
    --upsample False \
    --mongo_host localhost \
    --mongo_port 27017 \
    --learning_rate 0.001 \
    --layers ${LAYERS} \
    --lstm_dim 100 \
    --polysemy_dim 4 \
    --dense_dim 512 \
    --num_epochs 30 \
    --summary_record_freq 10
#    --usr_embeds_files ${USR_EMBED_FILES} \
