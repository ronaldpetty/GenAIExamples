#!/bin/bash

# eval dataset
FILEDIR=/mnt/disk3/minminhou/datasets/patronus_enterprise_pii/ #/localdisk/minminho/datasets/patronus_enterprise_pii/
FILENAME=patronus_enterprise_pii_eval_v2.csv
OUTPUT=nomic_lr_clf_v2_0p5_eval


# encoder embedding model related params
MODEL=nomic-ai/nomic-embed-text-v1 #BAAI/bge-large-en-v1.5
TOKENIZER=nomic-ai/nomic-embed-text-v1 #BAAI/bge-large-en-v1.5
MAXSEQLEN=4096
PREFIX=classification

# run time related params
BATCHSIZE=8
THRESHOLD=0.5

# logistic regression related
LRCLF=/mnt/disk3/minminhou/saved_models/enterprise_pii_lr_clf_v2.joblib #/localdisk/minminho/models/enterprise_pii_lr_clf_v1.joblib


python enterprise_pii_classifier_eval.py \
--filedir $FILEDIR \
--filename $FILENAME \
--output $OUTPUT \
--model $MODEL \
--tokenizer $TOKENIZER \
--max_seq_len $MAXSEQLEN \
--prefix $PREFIX \
--batch_size $BATCHSIZE \
--threshold $THRESHOLD \
--lr_clf $LRCLF \
--use_st_encoder