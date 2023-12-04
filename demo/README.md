## Magicoder Inference Demo

We provide the inference demo script for Magicoder models.
1. According to the instructions of [README](https://github.com/ise-uiuc/magicoder/blob/main/README.md), install the environment.
2. Install these packages:
```bash
pip install fire==0.5.0
pip install gradio==3.46.0
```
3. Run the command:
```bash
CUDA_VISIBLE_DEVICES=0 python magicoder_demo.py \
   --base_model "ise-uiuc/Magicoder-S-DS-6.7B" \
   --device "cuda:0" \
   --port 8080
```
