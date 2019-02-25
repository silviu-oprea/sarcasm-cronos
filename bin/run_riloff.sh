#!/usr/bin/env bash

if [[ "$#" -ne 5 ]]; then
    echo "Usage is $0 [dude] [batch size] [dropout rate] [layers:ia,lstm,conv,usr_embed] [usr embed files]"
    exit
fi

DUDE=$1
BATCH_SIZE=$2
DROPOUT_RATE=$3
LAYERS=$4
USR_EMBED_FILES=$5

python -u src/main.py \
    --tweets_col ${DUDE}_tweets \
    --buckets_col ${DUDE}_buckets \
    --train_buckets 5,0,8,1,4,9,6,2 \
    --valid_buckets 3 \
    --test_buckets 7 \
    --batch_size ${BATCH_SIZE} \
    --grad_clip_norm 0.1 \
    --dropout_rate ${DROPOUT_RATE} \
    --min_word_freq 2 \
    --min_doc_len 3 \
    --max_doc_len 40 \
    --word_embeds_col embeddings_glove \
    --word_embed_dim 100 \
    --upsample True \
    --mongo_host carahatas \
    --mongo_port 27017 \
    --learning_rate 0.001 \
    --layers ${LAYERS} \
    --lstm_dim 100 \
    --polysemy_dim 4 \
    --dense_dim 512 \
    --num_epochs 30 \
    --summary_record_freq 10 
#    --usr_embeds_files ${USR_EMBED_FILES} \
