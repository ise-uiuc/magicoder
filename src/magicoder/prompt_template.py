# Used to train the model
MAGICODER_PROMPT = """You are a helpful coding assistant that responds appropriately to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

# Used to format the `{instruction}` part above
SRC_INSTRUCT_INSTRUCTION_PROMPT = """Write a solution to the following coding problem:
{problem}"""

# Used to format src-instruct data points
SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""
