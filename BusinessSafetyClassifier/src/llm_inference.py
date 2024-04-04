# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
import json


def format_prompt(text, tokenizer, prompt_template):
    chat = [
                {"role": "user",
                "content":prompt_template.format(text=text)}
            ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt


def clean_output(string, random_choice=True):
    labels = ["Yes", "No"]
    for category in labels:
        if category.lower() in string.lower():
            return category
    # if the output string cannot be mapped to one of the categories, we either return "FAIL" or choose a random label
    if random_choice:
        return random.choice(labels)
    else:
        return "FAIL"

def parse_output(output, args = None):
    try:
        if args.vllm_offline == False:
            output = output.split('[/INST]')[-1].strip(" ").strip("</s>")
        output_dict = json.loads(output)
        # print(output_dict)
    except Exception as e:
        print(f"Parsing failed for output: {output}, Error: {e}")
        output_cl = clean_output(output, random_choice=False)
        output_dict = {"reason": "FAIL", "answer": output_cl}
    return output_dict


def convert_to_numeric_label(output):
    pred = output['answer']
    if "yes" in pred.lower():
        return 1
    elif "no" in pred.lower():
        return 0
    else:
        return 0
        
##########################################################
######################### vLLM #############################
#############################################################
def setup_vllm(args):  
    # initialize offline engine
    llm = LLM(
        model=args.model, 
        #download_dir =args.model_dir, 
        pipeline_parallel_size = 1,
        tensor_parallel_size = args.tp_size,)
        #dtype = 'half')

    return llm


def vllm_batched_offline_generation(args=None, llm=None, text = None, prompt_template=None):
    # ref: https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference

    sampling_params = SamplingParams(
        top_p=args.top_p, #0.90,
        temperature=args.temperature, #0.8,
        max_tokens=args.max_new_tokens, #128,
    )

    # make prompts
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = []
    for txt in text:
        prompts.append(format_prompt(txt, tokenizer, prompt_template))

    # batch generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # parse output
    predictions = []
    reasons = []
    for output in outputs:
        generated_text = output.outputs[0].text
        # print(generated_text)
        out_dict = parse_output(generated_text, args)
        predictions.append(convert_to_numeric_label(out_dict))
        reasons.append(out_dict['reason'])

    return predictions, reasons


#############################################################
############ TGI on Gaudi ###################################
#############################################################

def generate_with_tgi(args=None, text = None, labels = None):
    from huggingface_hub import InferenceClient
    from concurrent.futures import ThreadPoolExecutor
    #TGI runs continuous batching: https://github.com/huggingface/text-generation-inference/tree/main/router#simple-continuous-batching
    #Therefore sending multiple requests concurrently can leverage it.

    ############## Need to debug and uncomment these lines################
    ####################################################################
    # make prompts
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # prompts = []
    # for txt in text:
    #     prompts.append(format_prompt(txt, tokenizer, prompt_template))

    prompts = text

    # make samples
    # {'text': text, 'label': label}

    tgi_client = TgiClient(
        args.server_address, args.max_concurrent_requests
    )

    tgi_client.run_generation(
        prompts, args.max_new_tokens
    )

    # for input, output in zip(, ):
    #     print('input: ', input)
    #     print('output: ', output)
    #     print('='*50)
    
    # parse output
    predictions = []
    reasons = []
    for output in tgi_client._generated_text:
        out_dict = parse_output(output, args)
        predictions.append(convert_to_numeric_label(out_dict))
        reasons.append(out_dict['reason'])

    return tgi_client._input_text, predictions, reasons



# TGI client code is adapted from: 
# https://github.com/huggingface/tgi-gaudi/blob/habana-main/examples/tgi_client.py
import os
import statistics
import threading
import time
import tqdm
from typing import List

from huggingface_hub import InferenceClient


def except_hook(args):
    print(f"Thread failed with error: {args.exc_value}")
    os._exit(1)

threading.excepthook = except_hook


class TgiClient:
    def __init__(
        self,
        server_address: str,
        max_num_threads: int
    ) -> None:
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_num_threads)
        self._client = InferenceClient(server_address)

        self._ttft = []
        self._tpot = []
        self._generated_text = []
        self._input_text = []

    def run_generation(
        self,
        samples: List[str],
        max_new_tokens: int
    ) ->  None:
        """
        Run generation for every sample in dataset.
        Creates a separate thread for every sample.
        """
        threads: List[threading.Thread] = []
        for sample in tqdm.tqdm(samples):
            self._semaphore.acquire()
            threads.append(
                threading.Thread(
                    target=self._process_sample, args=[sample, max_new_tokens]
                )
            )
            threads[-1].start()
        for thread in threads:
            if thread is not None:
                thread.join()

    def _process_sample(
        self,
        sample: str,
        max_new_tokens: int,
        # streaming: bool,
    ) -> str:
        """
        Generates response stream for a single sample.
        Collects performance metrics.
        """
        timestamp = time.perf_counter_ns()
        response = self._client.text_generation(
            sample, max_new_tokens=max_new_tokens, stream=False, details=True
        )
        
        self._semaphore.release()
        self._generated_text.append(response.generated_text)
        self._input_text.append(sample)

    def print_performance_metrics(
        self,
        duration_s: float
    ) -> None:
        def line():
            print(32*"-")

        line()
        print("----- Performance  summary -----")
        line()
        print(f"Throughput: {sum(self._generated_tokens) / duration_s:.1f} tokens/s")
        print(f"Throughput: {len(self._generated_tokens) / duration_s:.1f} queries/s")
        line()
        print(f"First token latency:")
        print(f"\tMedian: \t{statistics.median(self._ttft)*1e-6:.2f}ms")
        print(f"\tAverage: \t{statistics.fmean(self._ttft)*1e-6:.2f}ms")
        line()
        print(f"Output token latency:")
        print(f"\tMedian: \t{statistics.median(self._tpot)*1e-6:.2f}ms")
        print(f"\tAverage: \t{statistics.fmean(self._tpot)*1e-6:.2f}ms")
        line()


######################################################################


def generate_text(model = None, tokenizer= None, prompt = None, generation_params = None):
    
    tokenized_prompt = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # print('prompt applied with template: \n', tokenizer.apply_chat_template(prompt, tokenize=False))

    tokenized_prompt = tokenized_prompt.to('cuda')
    encoded_output = model.generate(tokenized_prompt, **generation_params)
    # print(encoded_output)
    output = tokenizer.decode(encoded_output[0])
    print('generated output:\n', output)
    return output


def sequential_generate_with_model_generate(args=None, generation_params=None, text = None):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) #"mistralai/Mixtral-8x7B-Instruct-v.01"
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    predictions = []
    reasons =[]
    for t in tqdm(text):

        chat = [
            {"role": "user",
            "content":PROMPT.format(text=t)}
        ]

        # prompt_for_llm = tokenizer.apply_chat_template(chat, tokenize=False)

        output = generate_text(model, tokenizer, chat, generation_params)

        output = parse_output(output)

        predictions.append(convert_to_numeric_label(output))
        reasons.append(output['reason'])

        print("="*50)


    return predictions, reasons


def batch_generation_with_model_generate(args=None, generation_params=None, text = None):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) #"mistralai/Mixtral-8x7B-Instruct-v.01"
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    input_dataset = CustomDataset(text, tokenizer)
    input_dataloader = DataLoader(input_dataset, batch_size=args.batch_size)

    predictions = []
    reasons = []

    for batch in input_dataloader:
        inputs = tokenizer(batch, padding=True, return_tensors = 'pt').to(args.device)
        # batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)
            generated_txts = tokenizer.batch_decode(outputs)
            # print(generated_txts)

        for i in range(args.batch_size):
            txt = generated_txts[i]
            out_dict = parse_output(txt)
            predictions.append(convert_to_numeric_label(out_dict))
            reasons.append(out_dict['reason'])
    
    return predictions, reasons


def generate_with_pipeline(args = None, text= None, generation_params=None):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    inputs = CustomDataset(text, tokenizer)
    pipe = pipeline(
        "text-generation", 
        model = args.model, 
        device_map = "auto", 
        **generation_params)

    # Pipeline with tokenizer without pad_token cannot do batching.
    pipe.tokenizer.pad_token_id = tokenizer.eos_token_id

    for out in tqdm(pipe(inputs, batch_size = args.batch_size, **generation_params)):
        print(out)
        # for i in range(args.batch_size):
        #     print(out[i]['generated_text'])
        # generated_text_batch = out[0]['generated_text']
        # print(generated_text_batch)



def run_llm_inference_alternatives(args, text, prompt_template):
    generation_params = dict(
        top_p=args.top_p, #0.90,
        temperature=args.temperature, #0.8,
        max_new_tokens=args.max_new_tokens, #128,
        # return_full_text=False,
        do_sample=True,
        use_cache=False
    )
    if args.generate_with_pipeline == True:
        generate_with_pipeline(args, text, generation_params)

    elif args.batch_model_generate == True:
        predictions, reasons = batch_generation_with_model_generate(args, generation_params, text)

    elif args.vllm_offline == True:
        predictions, reasons = vllm_batched_offline_generation(args, text, prompt_template)
    elif args.tgi_concurrent == True:
        generate_with_tgi(args, text)
    else:
        predictions, reasons = sequential_generate_with_model_generate(args, generation_params, text)

    return predictions, reasons