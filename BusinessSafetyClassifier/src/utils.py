import pandas as pd
import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prompt_templates import PROMPT_BUSINESS_SENSITIVE

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filedir", type=str, default="", help="file directory where input and output are stored"
    )

    parser.add_argument(
        "--filename", type = str, default=""
    )

    parser.add_argument(
        "--output", type=str, default=""
    )

    parser.add_argument(
        "--model", type=str, default="", help="model to be used"
    )

    parser.add_argument(
        "--model_dir", type=str, default="", help="model directory to be used for vllm offline mode"
    )

    parser.add_argument(
        "--tokenizer", type=str, default="", help="tokenizer to be used"
    )

    parser.add_argument(
        "--prefix", type=str, default="", help="prefix for encoder"
    )

    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="max sequence length for embedding model"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32
    )

    parser.add_argument(
        "--threshold", type=float, default=0.5, help="threshold for classifier"
    )

    parser.add_argument(
        "--lr_clf", type=str, help="path to saved logistic regression classifier"
    )

    # parser.add_argument(
    #     "--use_m2_bert", action = "store_true"
    # )

    # parser.add_argument(
    #     "--use_st_encoder", action = "store_true"
    # )

    # parser.add_argument(
    #     "--use_decoder", action="store_true"
    # )


    # for logistic regression classifier training
    parser.add_argument(
        "--random_seed", type=int, default=123
    )

    parser.add_argument(
        "--max_iter", type=int, default=100
    )

    # for annotation with llm
    parser.add_argument(
        "--top_p", type = float, default=0.9
    )

    parser.add_argument(
        "--temperature", type = float, default=0.8, help="temperature for LLM generation"
    )

    parser.add_argument(
        "--max_new_tokens", type = int, default=128
    )

    # parser.add_argument(
    #     "--generate_with_pipeline", action = "store_true"
    # )

    # parser.add_argument(
    #     "--batch_model_generate", action = "store_true"
    # )

    parser.add_argument(
        "--vllm_offline", action = "store_true"
    )

    parser.add_argument(
        "--tgi_concurrent", action = "store_true"
    )

    parser.add_argument(
        "--optimum_habana", action = "store_true"
    )


    parser.add_argument(
        "--device", type = str, default='hpu', help="options: hpu, cuda, cpu"
    )

    parser.add_argument(
        "--run_prefilters", action = "store_true"
    )

    parser.add_argument(
        "--rerun_failed", action = "store_true"
    )
    

    parser.add_argument(
        "--tp_size", type=int, default=8, help="tensor parallel size"
    )

    parser.add_argument(
        "--run_eval", action = "store_true"
    )

    parser.add_argument(
        "--text_col", type = str, default='text', help="column name for text"
    )

    parser.add_argument(
        "--length_col", type = str, default='length', help="name of column that contains length of the text"
    )

    parser.add_argument(
        "--label_col", type = str, default='label', help="name of column that contains labels"
    )

    parser.add_argument(
        "--eval_size", type=int, default=300
    )

    parser.add_argument(
        "--max_concurrent_requests", type=int, default=256, help="Max number of concurrent requests"
    )

    parser.add_argument(
        "--server_address", type=str, default="http://localhost:8080", help="Address of the TGI server"
    )

    # args for text generation with optimum habana
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Whether to reuse key/value cache for decoding. It should save memory.",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        action="store_true",
        help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
    )
    parser.add_argument(
        "--attn_softmax_bf16",
        action="store_true",
        help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
        "is also running in lower precision.",
    )
    parser.add_argument(
        "--gaudi_lazy_mode",
        action="store_true",
        help="Whether to use lazy mode, should improve performance.",
    )

    args = parser.parse_args()
    return args

def add_prefix(text_batch, prefix):
    rt_text = [prefix+': '+t for t in text_batch]
    return rt_text

def load_data_for_st_encoders(args):
    df = pd.read_csv(args.filedir+args.filename)
    text = df[args.text_col].to_list()
    text = add_prefix(text, args.prefix)
    labels = df[args.label_col].to_list()
    return text, labels

def make_batches(text_list, bs):
    l = len(text_list)
    for i in range(0, l, bs):
        yield text_list[i:min(l, i+bs)]


def calculate_metrics(predictions, labels):
    assert len(labels) == len(predictions), "labels and predictions have different lengths"
    labels = np.array(labels)
    predictions = np.array(predictions)
    # accuracy
    acc = np.mean(labels == predictions)
    print('accuracy: {:.3f}'.format(acc))

    # true positive
    tp = np.sum((predictions ==1)&(labels==1))

    # false positive rate
    # find # of fp
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fpr = fp/(fp+tn)
    print('false positive rate: {:.3f}'.format(fpr))

    # false negative
    fn = np.sum((predictions==0)&(labels==1))

    # precision
    precision = tp/(tp+fp)
    # recall
    recall = tp/(tp+fn)
    print('precision: {:.3f}, recall: {:.3f}'.format(precision, recall))





class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, prompt_template):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        # output = pipe(prompt)[0]['generated_text'][-1]
        chat = [
            {"role": "user",
            "content":self.prompt_template.format(text=self.data[i])}
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return prompt




