# Reproduce the experiments

> [!WARNING]
> This documentation is still WIP. Raise an [issue](https://github.com/ise-uiuc/magicoder/issues) in case you found any errors.

In this document, we provide the instructions for reproducing the experiments in the paper.

> [!IMPORTANT]
> **General requirements**
>
> Before you start, make sure you cloned the respository.
> Here are the environment and hardware requirements to 100% reproduce the paper results.
>
> - Two NVIDIA A100 80G GPUs
> - Python 3.10.12
> - Having installed [pdm](https://pdm-project.org/latest/) and having set it up for the magicoder repo (e.g., `pdm install`).
> - Now you should have the same package versions as specified in [pdm.lock](/pdm.lock).

## Reproduce HumanEval(+) and MBPP(+)

We pack multiple problems into one batch to speed up the inference. A different batch size may lead to slightly worse/better results due to the floating point round off resulted from the underlying [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) optimization. We chose the batch size that can maximize the utilization of 1 or 2 GPUs depending on the resource availability at the time we ran the evaluation.

Make sure you set `CUDA_VISIBLE_DEVICES` to the 1 or 2 GPUs you want to use and `cd`ed to the root directory of the repo. Some larger batch sizes require 2 GPUs.

### HumanEval(+)

<details>

<summary>Magicoder-CL-7B</summary>

```bash
MODEL_KEY=codellama/CodeLlama-7b-Python-hf
MODEL=ise-uiuc/Magicoder-CL-7B
DATASET=humaneval
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl

python -m experiments.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 16 \
  --n_samples_per_problem 1 \
  --n_batches 1

evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
# humaneval (base tests)
# pass@1: 0.604
# humaneval+ (base + extra tests)
# pass@1: 0.555
```

</details>

<details>

<summary>Magicoder-S-CL-7B</summary>

```bash
MODEL_KEY=codellama/CodeLlama-7b-Python-hf
MODEL=ise-uiuc/Magicoder-S-CL-7B
DATASET=humaneval
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl

python -m experiments.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 16 \
  --n_samples_per_problem 1 \
  --n_batches 1

evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
# humaneval (base tests)
# pass@1: 0.707
# humaneval+ (base + extra tests)
# pass@1: 0.665
```

</details>

<details>

<summary>Magicoder-DS-6.7B</summary>

```bash
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-base
MODEL=ise-uiuc/Magicoder-DS-6.7B
DATASET=humaneval
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl

python -m experiments.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 28 \
  --n_samples_per_problem 1 \
  --n_batches 1

evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
# humaneval (base tests)
# pass@1: 0.665
# humaneval+ (base + extra tests)
# pass@1: 0.604
```

</details>

<details>

<summary>Magicoder-S-DS-6.7B</summary>

```bash
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-base
MODEL=ise-uiuc/Magicoder-S-DS-6.7B
DATASET=humaneval
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl

python -m experiments.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 28 \
  --n_samples_per_problem 1 \
  --n_batches 1

evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
# humaneval (base tests)
# pass@1: 0.768
# humaneval+ (base + extra tests)
# pass@1: 0.707
```

</details>

### MBPP(+)

Make sure you download the [EvalPlus repo](https://github.com/evalplus/evalplus) and performed `export PYTHONPATH=$EVALPLUS_REPO_ROOT`. We will use its `tools.sanitize` to sanitize the generated samples.

<details>

<summary>Magicoder-CL-7B</summary>

```bash
MODEL_KEY=codellama/CodeLlama-7b-Python-hf
MODEL=ise-uiuc/Magicoder-CL-7B
DATASET=mbpp
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl
SANITIZED_PATH=evalplus-$(basename $MODEL)-$DATASET-sanitized.jsonl

python -m experiments.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 24 \
  --n_samples_per_problem 1 \
  --n_batches 1

python -m tools.sanitize --dataset $DATASET --samples $SAVE_PATH
evalplus.evaluate --dataset $DATASET --samples $SANITIZED_PATH
# mbpp (base tests)
# pass@1: 0.642
# mbpp+ (base + extra tests)
# pass@1: 0.526
```

</details>

<details>

<summary>Magicoder-S-CL-7B</summary>

```bash
MODEL_KEY=codellama/CodeLlama-7b-Python-hf
MODEL=ise-uiuc/Magicoder-S-CL-7B
DATASET=mbpp
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl
SANITIZED_PATH=evalplus-$(basename $MODEL)-$DATASET-sanitized.jsonl

python -m experiments.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 24 \
  --n_samples_per_problem 1 \
  --n_batches 1

python -m tools.sanitize --dataset $DATASET --samples $SAVE_PATH
evalplus.evaluate --dataset $DATASET --samples $SANITIZED_PATH
# mbpp (base tests)
# pass@1: 0.684
# mbpp+ (base + extra tests)
# pass@1: 0.566
```

</details>

<details>

<summary>Magicoder-DS-6.7B</summary>

```bash
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-base
MODEL=ise-uiuc/Magicoder-DS-6.7B
DATASET=mbpp
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl
SANITIZED_PATH=evalplus-$(basename $MODEL)-$DATASET-sanitized.jsonl

python -m experiments.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 24 \
  --n_samples_per_problem 1 \
  --n_batches 1

python -m tools.sanitize --dataset $DATASET --samples $SAVE_PATH
evalplus.evaluate --dataset $DATASET --samples $SANITIZED_PATH
# mbpp (base tests)
# pass@1: 0.754
# mbpp+ (base + extra tests)
# pass@1: 0.619
```

</details>

<details>

<summary>Magicoder-S-DS-6.7B</summary>

```bash
MODEL_KEY=deepseek-ai/deepseek-coder-6.7b-base
MODEL=ise-uiuc/Magicoder-S-DS-6.7B
DATASET=mbpp
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl
SANITIZED_PATH=evalplus-$(basename $MODEL)-$DATASET-sanitized.jsonl

python -m experiments.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 24 \
  --n_samples_per_problem 1 \
  --n_batches 1

python -m tools.sanitize --dataset $DATASET --samples $SAVE_PATH
evalplus.evaluate --dataset $DATASET --samples $SANITIZED_PATH
# mbpp (base tests)
# pass@1: 0.757
# mbpp+ (base + extra tests)
# pass@1: 0.644
```

</details>

## Reproduce MultiPL-E

We use [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) for MultiPL-E evaluation.

## Reproduce DS-1000

Download [DS-1000 GitHub Repo](https://github.com/xlang-ai/DS-1000) and set the `PYTHONPATH` to the repo root. You would also need to tweek its source code to support the workflow. Then use the following command to perform DS-1000 generation:

```bash
python experiments/ds_1000.py \
    --dataset_path $PATH_TO_DS1000_DATA \
    --model_key $MODEL_KEY \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --mode $MODE \
```

After that, follow DS-1000 instructions to evaluate the generated samples.

## Reproduce data analysis

Here are some descriptions for the `experiments/data_embedding` directory:

- `length.py`: provides the token length distribution for data file problems and solutions.
- `cosine_similarity.py`: computes the cosine similarity between the TF-IDF embeddings of data file and HumanEval.
- `instruction_embedding.py`:  classifies and calculates the percentage composition of data within the data file based on the instruction you provide.

1. To depict the length distribution for either problems or solutions of the data file, you can run the command:

    ```bash
    python experiments/data_embedding/length.py 
    ```

    The result will be shown in `Length.png`

2. To see the similarity between the data file and HumanEval, you can run the command:

    ```bash
    python experiments/data_embedding/cosine_similarity.py
    ```

    The result will be shown in `HE_similarity_comparison.png`

3. To study the categories of the data file, there are two different modes:
    - In the **instruction** mode, the model will generate the corresponding embeddings according to the instructions and number of clusters you give, and then generate clusters based on these embeddings.

      You can change the clustering criteria by adjusting the `--instruction`.

      For example, if you want to cluster the data file according to the programming languages, you can run the command:

      ```bash
      python experiments/data_embedding/instructor_embedding.py \
      --data_files data-clean-decontaminated.jsonl \
      --model_key  instructor-base \
      --embedding_mode solution \
      --instruction "Represent the programming language used" \
      --n_clusters 2
      ```

      The clustering result will be shown in  `Clusters.png`.

    - In the **query** mode,  the model will generate the corresponding embeddings according to the instructions and queries you give,  then classifies them by calculating the cosine similarity between the embeddings of the data file and the embeddings of queries.

      You can change the classification criteria by adjusting the `--query_instruction` and `--queries`.

      For example, if you want to classify the data file according to the topic of the content, you can run the command:

      ```bash
      python experiments/data_embedding/instructor_embedding.py \
      --data_files data-clean-decontaminated.jsonl \
      --model_key  instructor-base \
      --embedding_mode solution \
      --instruction "Represent the code for retrieving" \
      --query_instruction "Represent the comment for retrieving the corresponding code" \
      --queries "Algorithmic and Data Structure Problems" "Mathematical and Computational Problems" "Database and SQL Problems" "System Design and Architecture Problems" "Security and Cryptography Problems" "Performance Optimization Problems" "Web Problems" "Domain Specific Problems" "User Interface and Application Design Problems" "Data Science and Machine Learning Problems" 
      ```

       The classification result will be shown in  `Pie_Chart.png`.
    - You can find more information about how to generate data embeddings by using specific instructions and queries [here](https://arxiv.org/pdf/2212.09741.pdf)

## Limitations

- In the evaluation of HumanEval(+) and MBPP(+), we did not consider the influence of randomness caused by the batch size choice. A different batch size can result in better/worse results due to the underlying [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) optimization.
- We primarily presented results from existing studies (e.g., [EvalPlus Leaderboard](https://evalplus.github.io)) and did not evaluate how varying prompts might impact the performance of Magicoder or other models.

In the near future, we will continue to improve Magicoder and provide more detailed and robust evaluations.
