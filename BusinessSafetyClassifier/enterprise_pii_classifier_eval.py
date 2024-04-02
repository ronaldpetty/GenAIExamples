from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
import pandas as pd
import numpy as np
from utils import get_args, load_data_for_st_encoders, make_batches
from utils import calculate_metrics



def load_and_tokenize_data(args):
    datafile = args.filedir + args.filename
    data = load_dataset(datafile).shuffle(seed=123).select(32)

    tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer,
    model_max_length=args.max_seq_len
    )

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_data = data.map(tokenize, batched=True)

    eval_dataloader = DataLoader(tokenized_data, batch_size=8)
    return eval_dataloader

def convert_similarity_scores_to_labels(cos_scores, args):
    labels = cos_scores>args.threshold
    return labels.squeeze().tolist()



def save_results(args, text, labels, predictions):
    df = pd.DataFrame({
        "text":text,
        "label":labels,
        "predictions": predictions
    })

    df.to_csv(args.filedir+args.output+'.csv')



def run_classification_use_st_encoder_similarity(args):
    prefix = args.prefix
    business_sensitive_description = "contain private financial numbers, \
        customer information, \
        personal performance information \
        information about sales or customer accounts"

    business_sensitive_description = prefix + ": "+business_sensitive_description

    model = SentenceTransformer(model_name_or_path=args.model, trust_remote_code=True) #, trust_remote_code = True)

    description_embedding = model.encode(business_sensitive_description, convert_to_tensor=True)

    eval_predictions = []

    eval_data, labels = load_data_for_st_encoders(args)

    # eval_data = ["Every morning, I make a cup of coffee to start my day.",
    #                   "hello world",
    #                   'Jack likes to eat apples',
    #                    'revenue growed 20%']
    

    for batch in make_batches(eval_data, args.batch_size):
        # batch = add_prefix(batch, args.prefix)
        print(batch)
        embeddings = model.encode(batch, convert_to_tensor=True)
        print('text embedding shape: ', embeddings.shape)
        cos_scores = st_util.cos_sim(embeddings, description_embedding)
        print('similarity shape: ', cos_scores.shape)
        # convert to binary label
        predictions = convert_similarity_scores_to_labels(cos_scores, args)
        eval_predictions.extend(predictions)
        print('='*50)
    
    calculate_metrics(eval_predictions, labels)
    save_results(args, eval_data, labels, eval_predictions)
    return eval_predictions


def run_classification_use_st_encoder_lr_clf(args):
    from joblib import load

    model = SentenceTransformer(model_name_or_path=args.model, trust_remote_code=True)

    lr_classifier = load(args.lr_clf)

    eval_predictions = []

    eval_data, labels = load_data_for_st_encoders(args)

    for batch in make_batches(eval_data, args.batch_size):
        # batch = add_prefix(batch, args.prefix)
        print(batch)
        embeddings = model.encode(batch, convert_to_tensor=True).cpu()
        print('text embedding shape: ', embeddings.shape)
        # prediction
        predictions = lr_classifier.predict(embeddings)
        # predictions = convert_similarity_scores_to_labels(predictions)
        eval_predictions.extend(predictions)
        print('='*50)
    
    calculate_metrics(eval_predictions, labels)
    save_results(args, eval_data, labels, eval_predictions)



def run_classification_use_m2_bert(input_dataloader, args):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels = 1,
        trust_remote_code=True)
  
    model.eval()
    eval_predictions = []
    with torch.no_grad():
        for batch in input_dataloader:
            outputs = model(**batch)
            print(outputs)
            predictions = outputs['logits'].squeeze().tolist()
            print(predictions)
            eval_predictions.extend(predictions)
        
    return eval_predictions


def main():
    args = get_args()
    # model_name = args.model #"togethercomputer/m2-bert-80M-8k-retrieval"
    # tokenizer_name = args.tokenizer #"bert-base-uncased"

    # max_seq_length = args.max_seq_len
    testing_string = [["Every morning, I make a cup of coffee to start my day.",
                      "hello world"],
                      ['Jack likes to eat apples',
                       'revenue growed 20%']]
    

    input_batch = testing_string
    if args.use_m2_bert == True:
        predictions = run_classification_use_m2_bert(input_batch, args)

    elif args.use_st_encoder == True:
        # predictions = run_classification_use_st_encoder_similarity(args)
        predictions = run_classification_use_st_encoder_lr_clf(args)

    print(predictions)


if __name__=="__main__":
    main()