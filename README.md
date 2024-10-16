# A Unified Subset Selection Framework for Efficient AI Evaluation


As the benchmark dataset of modern AI systems continue to scale, the computational and time costs of evaluation become increasingly prohibitive. These costs are not one-time investments; each time a model checkpoint is trained or fine-tuned, it must be retested with these benchmarks, leading to a costly and repetitive cycle. In some cases, evaluation costs may even surpass those of pretraining when evaluating checkpoints.

This repository implements an innovative item selection algorithm for efficient AI evaluation. Using Large Language Models (LLMs) as an example, below are the detailed steps and instructions for implementation.

## 1. Dataset Acquisition

We scrape datasets from the [HELM](https://crfm.stanford.edu/helm/classic/latest/) website, which records the performance of large models on leaderboard benchmarks. We select the **Classic** version of the dataset to maximize the number of items.

Steps for dataset acquisition:
- We download the response of each large model in `.json` format.
- Each data entry contains the following key information:
  - **Model ID**: A unique identifier corresponding to each model.
  - **Item ID**: A unique identifier for each item.
  - **Item Score**: The score is derived from HELM’s evaluation metrics. For instance, using the EM (Exact Match) metric, the score is binary (0 or 1), indicating whether the model answered the item correctly.

This process generates a large dataset of model responses, which serves as the foundation for training the item features (using Item Response Theory).

## 2. Benchmark Preprocessing 



The collected response data on benchmarks are used to estimate each item's features (e.g., difficulty). We use the IRT model and cross-entropy loss to fit the large-scale model response data, allowing for the estimation of item difficulties and discriminations.

To begin model training, execute the `training/launch.py` file. 

The trained model parameters will be saved in the `scripts/model` folder. After training, the model parameters are saved for use in the subsequent item selection phase.

## 3. Subset Selection

Once training is complete, run `main.py` for Subset Selection. By running the script, you can evaluate the effectiveness of the item selection algorithm and further refine it.


## Dependencies

- Python 3.7+

