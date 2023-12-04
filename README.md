# Magicoder: Source Code Is All You Need

This is the repo for our paper "Magicoder: Source Code Is All You Need"!

## News

- 🔥🔥🔥[2023/12/04] We have released our **Magicoder-CL-7B**, **Magicoder*S*-CL-7B**, **Magicoder-DS-6.7B**, and **Magicoder*S*-DS-6.7B**! Our **Magicoder*S*-DS-6.7B** model achieves the **76.8 pass@1** on the [HumanEval Benchmarks](https://github.com/openai/human-eval), which surpasses **DeepSeek-Coder-Instruct-6.7B** with ×8 fewer training tokens, outperforms **WizardCoder-CL-34B** and **ChatGPT-3.5**, and also closely matches **DeepSeek-Coder-Instruct-34B**!

|  Model  |  Checkpoint  | Size    | HumanEval (+) |   MBPP (+) | Demo | License |
| ----- |------| ---- |------|-------| ----- |  ----- | 
|  Magicoder-CL  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-CL-7B" target="_blank">HF Link</a>   |  7B  |  60.4 (55.5)   | 64.2 (52.6) | -- |  --  |
|  Magicoder*S*-CL  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-S-CL-7B" target="_blank">HF Link</a>   |  7B  |  70.7 (66.5)   | 68.4 (56.6) | -- |  --  |
|  Magicoder-DS  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-DS-6.7B" target="_blank">HF Link</a>   |  6.7B  |  66.5 (60.4)   | 75.4 (61.9) | -- |  --  |
|  Magicoder*S*-DS  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B" target="_blank">HF Link</a>   |  6.7B  |  **76.8** (**70.7**)   | **75.7** (**64.4**) | -- |  --  |

- 🔥🔥🔥[2023/12/04] We have released our Magicoder datasets: [**Magicoder_oss_instruct_75k**](https://huggingface.co/datasets/ise-uiuc/Magicoder_oss_instruct_75k) and [**Magicoder_evol_instruct_110k**](https://huggingface.co/datasets/ise-uiuc/Magicoder_evol_instruct_110k)!


## Overview

TODO

## Setup and Demo

> [!IMPORTANT]
> **Setting up developing environment**
> - Clone and `cd` into the repo, then run:
> - `pdm install` or `pip install -e .`

You can find an inference demo of Magicoder [here](https://github.com/ise-uiuc/magicoder/tree/main/demo).


## Data generation

Make sure you have set up your `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. Then run with

```bash
python src/magicoder/generate_data.py \
  --seed_code_start_index ${START_INDEX_OF_RAW_DATA} \
  --max_new_data ${MAX_DATA_TO_GENERATE}
```

To continue an interrupted run, use `--continue_from` flag:

```bash
python src/magicoder/generate_data.py \
  --seed_code_start_index ${START_INDEX_OF_RAW_DATA} \
  --max_new_data ${MAX_DATA_TO_GENERATE} \
  --continue_from ${PATH_TO_DATA_FILE}
```

## Fine-tuning

TODO

## Citation

TODO
