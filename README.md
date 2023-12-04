# 🎩 Magicoder: Source Code Is All You Need

<p align="center">
    <a href="https://arxiv.org/abs/1234.5678"><img src="https://img.shields.io/badge/arXiv-1234.5678-b31b1b.svg"></a>
    <a href="https://github.com/ise-uiuc/magicoder/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://huggingface.co/ise-uiuc"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm.svg"></a>
</p>

[jw: add toc after the sections are ready]

## About

[jw: add a few lines of desc on what we have and highlight/prioritize what's new]

## Magicoder Models

|  Model  |  Checkpoint  | Size    | HumanEval (+) |   MBPP (+) | Demo | License |
| ----- |------| ---- |------|-------| ----- |  ----- | 
|  Magicoder-CL  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-CL-7B" target="_blank">HF Link</a>   |  7B  |  60.4 (55.5)   | 64.2 (52.6) | -- |  --  |
|  Magicoder*S*-CL  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-S-CL-7B" target="_blank">HF Link</a>   |  7B  |  70.7 (66.5)   | 68.4 (56.6) | -- |  --  |
|  Magicoder-DS  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-DS-6.7B" target="_blank">HF Link</a>   |  6.7B  |  66.5 (60.4)   | 75.4 (61.9) | -- |  --  |
|  Magicoder*S*-DS  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B" target="_blank">HF Link</a>   |  6.7B  |  **76.8** (**70.7**)   | **75.7** (**64.4**) | -- |  --  |

## OSS-Instruct Dataset

- [**Magicoder_oss_instruct_75k**](https://huggingface.co/datasets/ise-uiuc/Magicoder_oss_instruct_75k)
- [**Magicoder_evol_instruct_110k**](https://huggingface.co/datasets/ise-uiuc/Magicoder_evol_instruct_110k)

## Quick Start

[jw: inline the demo instead of redirecting it to a new link. put most things into code. add some concise desc.]

```bash
git clone https://github.com/ise-uiuc/magicoder.git
cd magicoder
pdm install
python magicoder_demo.py --base_model "ise-uiuc/Magicoder-S-DS-6.7B" \
                         --device "cuda:0" --port 8080
```


## Generating Synthetic Data with OSS-Instruct

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

## Fine-tuning over OSS-Instruct Datasets

TODO

## Citation

TODO
