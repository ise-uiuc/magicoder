from transformers import AutoTokenizer
import transformers
import os
import sys
import fire
import torch
import gradio as gr


def main(
    base_model="ise-uiuc/Magicoder-S-DS-6.7B",
    device="cuda:0",
    port=8080,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=base_model,
        torch_dtype=torch.float16,
        device=device
    )
    def evaluate_magicoder(
        instruction,
        temperature=1,
        max_new_tokens=2048,
    ):
        MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
""" 
        prompt = MAGICODER_PROMPT.format(instruction=instruction)

        if temperature > 0:
            sequences = pipeline(
                prompt,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        else:
            sequences = pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
            )
        for seq in sequences:
            print('==========================question=============================')
            print(prompt)
            generated_text = seq['generated_text'].replace(prompt, "")
            print('===========================answer=============================')
            print(generated_text)
            return generated_text

    gr.Interface(
        fn=evaluate_magicoder,
        inputs=[
            gr.components.Textbox(
                lines=3, label="Instruction", placeholder="Anything you want to ask Magicoder ?"
            ),
            gr.components.Slider(minimum=0, maximum=1, value=1, label="Temperature"),
            gr.components.Slider(
                minimum=1, maximum=2048, step=1, value=512, label="Max tokens"
            ),
        ],
        outputs=[
            gr.components.Textbox(
                lines=30,
                label="Output",
            )
        ],
        title="Magicoder",
        description="This is a LLM playground for Magicoder! Follow us on Github: https://github.com/ise-uiuc/magicoder and Huggingface: https://huggingface.co/ise-uiuc."
    ).queue().launch(share=True, server_port=port)

if __name__ == "__main__":
    fire.Fire(main)
