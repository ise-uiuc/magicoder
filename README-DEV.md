# Implementation Details of 🎩Magicoder

> [!WARNING]
> This document is a work in progress. It is not yet complete.

## Data Generation

Make sure you have set up your `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. Then run with

```bash
python src/magicoder/generate_data.py \
  --seed_code_start_index ${START_INDEX_OF_RAW_DATA} \
  --max_new_data ${MAX_DATA_TO_GENERATE} \
  --data_dir python \
  --tag python
```

To continue an interrupted run, use `--continue_from` flag:

```bash
python src/magicoder/generate_data.py \
  --seed_code_start_index ${START_INDEX_OF_RAW_DATA} \
  --max_new_data ${MAX_DATA_TO_GENERATE} \
  --data_dir python \
  --continue_from ${PATH_TO_DATA_FILE}
```
