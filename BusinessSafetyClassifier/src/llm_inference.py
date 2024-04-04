from vllm import LLM, SamplingParams
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


def generate_with_tgi(args=None, text = None):
    from huggingface_hub import InferenceClient
    from concurrent.futures import ThreadPoolExecutor
    #TGI runs continuous batching: https://github.com/huggingface/text-generation-inference/tree/main/router#simple-continuous-batching
    #Therefore sending multiple requests concurrently can leverage it.

    # make prompts
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = []
    for txt in text:
        prompts.append(format_prompt(txt, tokenizer))


    client = InferenceClient(model="http://127.0.0.1:8080")

    def gen_text(text):
        return client.text_generation(text,max_new_tokens=256)

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor: 
        out = list(executor.map(gen_text, prompts))

    print(out)


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