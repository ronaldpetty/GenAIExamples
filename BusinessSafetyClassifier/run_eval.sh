#!/bin/bash

# eval dataset
FILEDIR=/mnt/disk3/minminhou/datasets/patronus_enterprise_pii/
FILENAME=annotated_patronus_enterprise_pii_eval.csv
OUTPUT=eval_results_patronus_enterprise_pii


# encoder embedding model related params
MODEL=nomic-ai/nomic-embed-text-v1
TOKENIZER=nomic-ai/nomic-embed-text-v1
PREFIX=classification

# run time related params
BATCHSIZE=8

# Column names for text and label columns
TEXT=text
LABEL=label

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
--text_col $TEXT \
--label_col $LABEL \

