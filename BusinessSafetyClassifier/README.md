# Business Safety Classifier

## Overview
This folder contains a tested recipe for training a Business Safety Classifier with custom data. No human annotation is required. However, a few tens of human annotations are needed if you want to get a sense of the accuracy of the LLM annotations. We provide evaluation code to show accuracy of LLM annotations by comparing to human annotations as gold labels.

### Use cases of Business Safety classifier
Enterprise applications often require the output content from LLMs or RAG systems to be free of business senstive information such as private financial figures, sales accounts and confidential customer information. Another source of sensitive information that enterprises may want to exclude from LLMs and RAG output is human resources data such as employee performance reviews. A binary classifier that can detect such sensitive content is imperative in protecting business safety of enterprises running Gen AI applications.

### Overall workflow of this recipe
1. Annotate dataset using a LLM (evaluation of LLM annotations is optional)
2. Train a logistic-regression classifier using LLM-annotated data
3. Evaluate the logistic-regression classifier

### Validated models and dataset
We have validated this recipe with the following models.
Annotator LLM: ```mistralai/Mixtral-8x7B-Instruct-v0.1```
Embedding model: ```nomic-ai/nomic-embed-text-v1```

We showcase our recipe with the open-sourced [Patronus EnterprisePII dataset](https://www.patronus.ai/announcements/patronus-ai-launches-enterprisepii-the-industrys-first-llm-dataset-for-detecting-business-sensitive-information), which is used in MosaicML's LLM Eval Gauntlet and the [Enterprise Scenarios Leaderboard](https://huggingface.co/blog/leaderboard-patronus) on Huggingface.

## Prerequisites

Before running the scripts, make sure you have the following installed:

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone this repository:

    ```shell
    git clone 
    ```

2. Navigate to the project directory:

    ```shell
    cd BusinessSafetyClassifier
    ```

3. Install the required Python packages:

    ```shell
    pip install -r requirements.txt
    ```

## Annotate unlabeled dataset with LLM

 Follow the steps below to annotate the Patronus EnterprisePII dataset with `mistralai/Mixtral-8x7B-Instruct-v0.1`. We also provide options to customize the recipe, see the [How to customize](#how-to-customize) section.


1. Download the dataset
We will need to get the Patronus EnterprisePII dataset from the llm-foundry repo. You can either clone the llm-foundry repo and then copy the dataset from the repo to your data folder as shown below, or you can directly download the file by click on the "Download raw file" button.

    ```shell
    git clone https://github.com/mosaicml/llm-foundry.git
    cp llm-foundry/scripts/eval/local_data/safety/enterprise_pii_classification.jsonl /path/to/your/data/folder/
    ```

2. Preprocess the dataset
The preprocessing step is specific to the Patronus EnterprisePII dataset where we get the actual text components and the gold labels from the original jsonl file. </br>
Specify the `FILEDIR`, `FILENAME` and `OUTPUT` variables in the `run_process_enterprisepii_dataset.sh` script and then run the command below. After the script is successfully completed, you will get three csv files: 1) the whole dataset, 2) randomly sampled training set, 3) test set that is exclusive of the training set.

    ```shell
    bash run_process_enterprisepii_dataset.sh
    ```

3. Run the annotation script
Note: Currently we use vllm's offline batch mode to run text generation on NV GPUs. We will enable this function on Gaudi platform in near future. 

    ```shell
    bash run_annotation.sh
    ```
After the script is successfully completed, you will get a csv file with LLM's predicted labels in it. For the Patronus EnterprisePII dataset, we also enabled calculation of annotation accuracy. You should see metrics printed out that are similar to the ones listed below. Since there is randomness in LLM generation, you may not see exactly the same numbers. Randomness is introduced as to allow for re-generation of annotations if the annotation failed in the first round.

|Accuracy|0.909|
|Precision|0.883|
|Recall|0.940|


## How to customize