# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from transformers import pipeline, LlamaForCausalLM, LlamaTokenizerFast
from cog import BasePredictor, Input, Path


MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
{instruction}
@@ Response
"""

css = """
#q-output {
    max-height: 60vh;
    overflow: auto;
}
"""


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        model_id = "ise-uiuc/Magicoder-S-DS-6.7B"
        local_files_only = True  # set to True if model is cached
        cache_dir = "model_cache"
        model = LlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def predict(
        self,
        instruction: str = Input(
            description="Input instruction.",
            default="Write a snake game in Python using the turtle library (the game is created by Magicoder).",
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, 1 is random and 0 is deterministic.",
            ge=0,
            le=1,
            default=0,
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens.",
            default=2048,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        prompt = MAGICODER_PROMPT.format(instruction=instruction)
        result = self.generator(
            prompt,
            max_length=max_tokens,
            num_return_sequences=1,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
        )
        return result[0]["generated_text"].replace(prompt, "")
