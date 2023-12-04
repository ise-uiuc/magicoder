# Magicoder

> [!IMPORTANT]
> **Setting up developing environment**
> - Clone and `cd` into the repo, then run:
> - `pdm install` or `pip install -e .`

## News

|  Model  |  Checkpoint  | Size    | HumanEval (+) |   MBPP (+) | Demo | License |
| ----- |------| ---- |------|-------| ----- |  ----- | 
|  Magicoder-CL  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-CL-7B" target="_blank">HF Link</a>   |  7B  |  60.4 (55.5)   | 64.2 (52.6) | -- |  --  |
|  Magicoder*S*-CL  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-S-CL-7B" target="_blank">HF Link</a>   |  7B  |  70.7 (66.5)   | 68.4 (56.6) | -- |  --  |
|  Magicoder-DS  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-DS-6.7B" target="_blank">HF Link</a>   |  6.7B  |  66.5 (60.4)   | 75.4 (61.9) | -- |  --  |
|  Magicoder*S*-DS  |   🤗 <a href="https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B" target="_blank">HF Link</a>   |  6.7B  |  **76.8** (**70.7**)   | **75.7** (**64.4**) | -- |  --  |


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
