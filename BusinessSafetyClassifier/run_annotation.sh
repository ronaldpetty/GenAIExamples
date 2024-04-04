#!/bin/bash

# eval dataset
FILEDIR=/mnt/disk3/minminhou/datasets/patronus_enterprise_pii/
FILENAME=processed_patronus_enterprise_pii.csv
OUTPUT=annotated_patronus_enterprise_pii

#model
MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
TOKENIZER=mistralai/Mixtral-8x7B-Instruct-v0.1
MAXNEWTOKEN=256
# MODELDIR=/mnt/disk3/minminhou/huggingface/transformers/

# tensor parallel for vllm
TP=4

# train test split
TESTSIZE=300

python src/annotate_data_with_llm.py \
--filedir $FILEDIR \
--filename $FILENAME \
--output $OUTPUT \
--model $MODEL \
--tokenizer $TOKENIZER \
--max_new_tokens $MAXNEWTOKEN \
--tp_size $TP \
--eval_size $TESTSIZE \
--run_prefilters \
--vllm_offline \
--rerun_failed \
--run_eval



