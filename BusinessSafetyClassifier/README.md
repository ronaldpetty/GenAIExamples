# Business Safety Classifier

## Overview
This folder contains a tested recipe for training a Business Safety Classifier with custom data. No human annotation is required. However, if you want to get a sense of the accuracy of the LLM annotations, a few tens of human annotations are recommended. We provide evaluation code to show accuracy, precision and recall of LLM annotations by taking human annotations as gold labels.

The overall workflow of this recipe is the following:
1. Annotate dataset using a LLM (evaluation of LLM annotations is optional)
2. Train a logistic-regression classifier using LLM-annotated data
3. Evaluate the logistic-regression classifier

Models used in our tested recipe:
In our recipe, we use Mixtral as the annotator LLM. We use 

## Prerequisites

Before running the scripts, make sure you have the following installed:

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone this repository:

    ```shell
    git clone https://github.com/your-username/BusinessSafetyClassifier.git
    ```

2. Navigate to the project directory:

    ```shell
    cd BusinessSafetyClassifier
    ```

3. Install the required Python packages:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

To run the scripts, follow these steps:

1. Step 1

    ```shell
    command 1
    ```

2. Step 2

    ```shell
    command 2
    ```

3. Step 3

    ```shell
    command 3
    ```

## Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).