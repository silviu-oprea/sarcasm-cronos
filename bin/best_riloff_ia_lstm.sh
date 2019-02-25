#!/usr/bin/env bash

python -u src/main.py \
    --tweets_col riloff_tweets \
    --buckets_col riloff_buckets \
    --train_buckets 0,1,2,3,4,5,6,7 \
    --valid_buckets 8 \
    --test_buckets 9 \
    --batch_size 16 \
    --grad_clip_norm 0.2 \
    --dropout_rate 0.5 \
    --min_word_freq 2 \
    --min_doc_len 3 \
    --max_doc_len 40 \
    --word_embeds_col embeddings_glove \
    --word_embed_dim 100 \
    --usr_embeds_files ''\
    --upsample False \
    --mongo_host localhost \
    --mongo_port 27017 \
    --learning_rate 0.001 \
    --layers ia,lstm \
    --lstm_dim 100 \
    --polysemy_dim 4 \
    --dense_dim 100 \
    --num_epochs 100 \
    --summary_record_freq 10
