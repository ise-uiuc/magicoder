# Magicoder

> [!IMPORTANT]
> **Setting up developing environment**
> - Clone and `cd` into the repo, then run:
> - `pdm install` or `pip install -e .`

## Data generation

Make sure you have set up your `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. Then run with

```bash
python src/magicoder/generate_data.py \
  --seed_code_start_index ${START_INDEX_OF_RAW_DATA} \
  --max_new_data ${MAX_DATA_TO_GENERATE}
```
