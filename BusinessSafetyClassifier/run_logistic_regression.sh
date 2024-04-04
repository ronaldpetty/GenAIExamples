#!/bin/bash

# dataset
FILEDIR=/mnt/disk3/minminhou/datasets/patronus_enterprise_pii/
FILENAME=annotated_patronus_enterprise_pii_train.csv

# Column names for text and label columns
TEXT=text
LABEL=final_prediction

# path for saving logistic regression classifier
OUTPUT=/mnt/disk3/minminhou/saved_models/lr_clf.joblib

# embedding model
MODEL=nomic-ai/nomic-embed-text-v1
BATCHSIZE=32

python src/train_logistic_regression_classifier.py \
--filedir $FILEDIR \
--filename $FILENAME \
--lr_clf $OUTPUT \
--model $MODEL \
--text_col $TEXT \
--label_col $LABEL \
--batch_size $BATCHSIZE