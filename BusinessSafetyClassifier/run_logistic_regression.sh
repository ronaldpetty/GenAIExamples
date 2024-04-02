#!/bin/bash

# dataset
FILEDIR=/mnt/disk3/minminhou/datasets/patronus_enterprise_pii/
FILENAME=patronus_enterprise_pii_train_v2.csv

# path for saving logistic regression classifier
OUTPUT=/mnt/disk3/minminhou/saved_models/test_lr_clf.joblib

# embedding model
MODEL=nomic-ai/nomic-embed-text-v1

python src/train_logistic_regression_classifier.py \
--filedir $FILEDIR \
--filename $FILENAME \
--lr_clf $OUTPUT \
--model $MODEL \