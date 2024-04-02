#!/bin/bash

# eval dataset
FILEDIR=/mnt/disk3/minminhou/datasets/patronus_enterprise_pii/ #/localdisk/minminho/datasets/patronus_enterprise_pii/
FILENAME=patronus_enterprise_pii_train_v2.csv
OUTPUT=/mnt/disk3/minminhou/saved_models/enterprise_pii_lr_clf_v2.joblib #/localdisk/minminho/models/enterprise_pii_lr_clf_v1.joblib


# model related params
MODEL=nomic-ai/nomic-embed-text-v1

python train_logistic_regression_classifier.py \
--filedir $FILEDIR \
--filename $FILENAME \
--output $OUTPUT \
--model $MODEL \