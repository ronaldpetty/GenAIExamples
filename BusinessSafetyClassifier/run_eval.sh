#!/bin/bash

# eval dataset
FILEDIR=/mnt/disk3/minminhou/datasets/patronus_enterprise_pii/
FILENAME=patronus_enterprise_pii_eval_v2.csv
OUTPUT=test_eval


# encoder embedding model related params
MODEL=nomic-ai/nomic-embed-text-v1
TOKENIZER=nomic-ai/nomic-embed-text-v1
# MAXSEQLEN=4096
PREFIX=classification

# run time related params
BATCHSIZE=8
# THRESHOLD=0.5

# logistic regression classifier path
LRCLF=/mnt/disk3/minminhou/saved_models/test_lr_clf.joblib


python src/enterprise_pii_classifier_eval.py \
--filedir $FILEDIR \
--filename $FILENAME \
--output $OUTPUT \
--lr_clf $LRCLF \
--model $MODEL \
--tokenizer $TOKENIZER \
--prefix $PREFIX \
--batch_size $BATCHSIZE \

