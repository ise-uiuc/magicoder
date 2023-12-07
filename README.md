# 🎩 Magicoder: Source Code Is All You Need

<p align="left">
    <a href="https://arxiv.org/abs/2312.02120"><img src="https://img.shields.io/badge/arXiv-2312.02120-b31b1b.svg?style=for-the-badge">
    <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge">
    <a href="https://huggingface.co/ise-uiuc/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ise--uiuc-%23ff8811.svg?style=for-the-badge">
</p>

<p align="left">
    🎩&nbsp;<a href="#-models">Models</a>
    | 📚&nbsp;<a href="#-dataset">Dataset</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 👀&nbsp;<a href="#-demo">Demo</a>
    | 📝&nbsp;<a href="#-citation">Citation</a>
    | 🙏&nbsp;<a href="#-acknowledgements">Acknowledgements</a>
</p>

> [!IMPORTANT]
> We are keeping improving the documents and adding more implementation details. Please stay tuned!

## About

* 🎩**Magicoder** is a model family empowered by 🪄**OSS-Instruct**, a novel approach to enlightening LLMs with open-source code snippets for generating *low-bias* and *high-quality* instruction data for code.
* 🪄**OSS-Instruct** mitigates the *inherent bias* of the LLM-synthesized instruction data by empowering them with *a wealth of open-source references* to produce more diverse, realistic, and controllable data.

![Overview of OSS-Instruct](assets/overview.svg)
![Overview of Result](assets/result.png)

## 🎩 Models

| Model                 | Checkpoint                                                        | Size | HumanEval (+)       | MBPP (+)            | Demo | License                                                                           |
|-----------------------|-------------------------------------------------------------------|------|---------------------|---------------------|------|-----------------------------------------------------------------------------------|
| Magicoder-CL-7B       | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-CL-7B)     | 7B   | 60.4 (55.5)         | 64.2 (52.6)         | --   | [Llama2](https://ai.meta.com/llama/license/)                                      |
| Magicoder-*S*-CL-7B   | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-S-CL-7B)   | 7B   | 70.7 (66.5)         | 68.4 (56.6)         | --   | [Llama2](https://ai.meta.com/llama/license/)                                      |
| Magicoder-DS-6.7B     | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-DS-6.7B)   | 6.7B | 66.5 (60.4)         | 75.4 (61.9)         | --   | [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL) |
| Magicoder-*S*-DS-6.7B | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B) | 6.7B | **76.8** (**70.7**) | **75.7** (**64.4**) | [Demo](https://67cc8c194b67d37b94.gradio.live)*  | [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL) |

<sub>*Demo link will expire on 12/8/2023.</sub>

## 📚 Dataset

* [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder_oss_instruct_75k): generated through **OSS-Instruct** using `gpt-3.5-turbo-1106` and used to train both Magicoder and Magicoder-S series.
* [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder_evol_instruct_110k): decontaminated and redistributed from [theblackcat102/evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1), used to further finetune Magicoder series and obtain Magicoder-S models.

## 🚀 Quick Start

```python
from transformers import pipeline
import torch

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

instruction = "Implement a high-level API for a TODO list application. The API takes as input an operation request and updates the TODO list in place. If the request is invalid, raise an exception."

prompt = MAGICODER_PROMPT.format(instruction=instruction)
generator = pipeline(
    model="ise-uiuc/Magicoder-S-DS-6.7B",
    task="text-generation",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
result = generator(prompt, max_length=2048, num_return_sequences=1, temperature=0.0)
print(result[0]["generated_text"])
```

This code snippet will generate the following output:

``````
Here is a simple Python implementation of a TODO list API:

```python
class TodoList:
    def __init__(self):
        self.todo_list = []

    def add_task(self, task):
        if not isinstance(task, str):
            raise ValueError("Task must be a string")
        self.todo_list.append(task)

    def remove_task(self, task):
        if task not in self.todo_list:
            raise ValueError("Task not found in the list")
        self.todo_list.remove(task)

    def get_tasks(self):
        return self.todo_list

    def update_task(self, old_task, new_task):
        if old_task not in self.todo_list:
            raise ValueError("Old task not found in the list")
        if not isinstance(new_task, str):
            raise ValueError("New task must be a string")
        index = self.todo_list.index(old_task)
        self.todo_list[index] = new_task

    def clear_list(self):
        self.todo_list = []
```

This API allows you to add tasks, remove tasks, get all tasks, update tasks, and clear the list. It also raises exceptions for invalid operations.

You can use this API like this:

```python
todo = TodoList()
todo.add_task("Buy groceries")
todo.add_task("Finish project")
print(todo.get_tasks())  # Output: ['Buy groceries', 'Finish project']
todo.update_task("Buy groceries", "Buy fruits")
print(todo.get_tasks())  # Output: ['Buy fruits', 'Finish project']
todo.remove_task("Finish project")
print(todo.get_tasks())  # Output: ['Buy fruits']
todo.clear_list()
print(todo.get_tasks())  # Output: []
```
``````

## 👀 Demo

We follow [WizardCoder](https://github.com/nlpxucan/WizardLM/blob/main/demo/wizardLM_demo.py) and provide the script to build a local demo server with [gradio](https://www.gradio.app). Refer to [/demo](demo/README.md) for more information.

Here are some interesting examples showing **Magicoder**'s improvements over base models:

<details>
<summary> <strong> Magicoder's Understanding Ability </strong> </summary>

We create the following two examples by making two original HumanEval problems more complicated. While both **Magicoder-*S*-DS-6.7B** and **deepseek-coder-6.7b-base**, which is **Magicoder**'s base model, can solve the original HumanEval problem, only our **Magicoder-*S*-DS-6.7B** can solve the new complicated problems. 

### Example 1
Original Problem: 
```
Write a function to, given list of integers, return list in "strange" order. "Strange" sorting, is when you start with the minimum value, then maximum of the remaining integers, then minimum and so on.
```

New Problem:
```
Write a function to, given list of integers, return list in "ascending" order. "Ascending" sorting, is when you start with the minimum value, then maximum of the remaining integers, then minimum and so on.
```

This problem is challenging because we change the name of a new way (not ascending) to sort integers from "strange" to "ascending". The model should understand that the word "ascending" here has a new meaning based on both the context and the fact that it is surrounded by quotation marks. As is shown in the following responses to the new problem from two models, **Magicoder-*S*-DS-6.7B** successfully understands this complicated requirement, while **deepseek-coder-6.7b-base** is misled and sort integers in ascending order instead.

Response to New Problem:
```python
# Magicoder-S-DS-6.7B (Correct)
def ascending_sort(lst):
    sorted_lst = []
    while lst:
        min_val = min(lst)
        sorted_lst.append(min_val)
        lst.remove(min_val)
        if lst:
            max_val = max(lst)
            sorted_lst.append(max_val)
            lst.remove(max_val)
    return sorted_lst

# deepseek-coder-6.7b-base (Wrong)
def sort_ascending(lst):
    lst.sort()       
    return lst
```

### Example 2
Original Problem: 
```
Write a function that takes an integer a and returns True if this ingeger is a cube of some integer number. Note: you may assume the input is always valid.
```

New Problem:
```
Write a function that takes an integer a and returns True if this ingeger is a cube of some integer number. Note: you should check whether the input is valid.
```

This problem is challenging because we ask the model to check the inputs' validity rather than assuming the input is always valid. While **Magicoder-*S*-DS-6.7B** successfully check the validity of the input, **deepseek-coder-6.7b-base** wrongly sets `a < 0` as the criterion of invalidity and thus fails to solve the problem.

Response to New Problem:
```python
# Magicoder-S-DS-6.7B (Correct)
def is_cube(a):
    if not isinstance(a, int):
        return False
    if a < 0:
        a = -a
    return round(a ** (1. / 3)) ** 3 == a

# deepseek-coder-6.7b-base (Wrong)
def is_cube(a):
    if a < 0:
        return False
    else:
        for i in range(1, a):
            if i**3 == a:
                return True
        return False
```

</details>



<details>
<summary> <strong> Magicoder's Ability to Use External Libraries </strong> </summary>

We create the following example that requires models to use external libraries for the certain task. While our **Magicoder-*S*-DS-6.7B** successfully follow the instruction in the example, **deepseek-coder-6.7b-base** tend to miss some requirements in the instruction.

Prompt:
```
Write a gradio application for the following use case: Take an input image and return a 45 degree clockwise rotated image. You should also add text description under the output showing the rotation degree.
```

This instruction is challenging because our **Magicoder**'s fine-tuning dataset **does not** contain the library "gradio" that is necessary for this task. Here are the gradio applications that **Magicoder-*S*-DS-6.7B** and **deepseek-coder-6.7b-base** construct respectively:

- **Magicoder-*S*-DS-6.7B**: **Correct!** It successfully performs the 45-degree rotation on the input image in the **clockwise** direction. As required in the instruction, it **adds the text description** under the output.

Interface:
![Magicoder](assets/magicoder-s-ds.png)


- **Deepseek-coder-6.7b-base**: Wrong. It wrongly performs the 45-degree rotation on the input image in the **counterclockwise** direction. Even worse, it **misses the text description** under the output.

Interface:
![deepseek-coder-6.7b-base](assets/ds-coder-base.png)

</details>


## 📝 Citation

```bibtex
@misc{magicoder,
    title={Magicoder: Source Code Is All You Need}, 
    author={Yuxiang Wei and Zhe Wang and Jiawei Liu and Yifeng Ding and Lingming Zhang},
    year={2023},
    eprint={2312.02120},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## 🙏 Acknowledgements

- [WizardCoder](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder): Evol-Instruct
- [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder): Base model for Magicoder-DS
- [CodeLlama](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/): Base model for Magicoder-CL
- [StarCoder](https://arxiv.org/abs/2305.06161): Data decontamination

## ⚠️ Important Note

- **Bias, Risks, and Limitations:** Magicoders may sometimes make errors, produce misleading contents, or struggle to manage tasks that are not related to coding.

- **Usage:** Magicoder models are trained on the synthetic data generated by OpenAI models. Please pay attention to OpenAI's [terms of use](https://openai.com/policies/terms-of-use) when using the models and the datasets. Magicoders will not compete with any OpenAI's commercial product.

## ⭐️ Star History

<a href="https://star-history.com/#ise-uiuc/magicoder&Timeline">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ise-uiuc/magicoder&type=Timeline&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ise-uiuc/magicoder&type=Timeline" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ise-uiuc/magicoder&type=Timeline" />
  </picture>
</a>

