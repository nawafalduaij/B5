# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Day 1a - Prompting with the Gemini API
#
# This tutorial introduces you to the fundamentals of working with the Gemini API, including prompt engineering techniques and code generation.
#
# ## Learning Objectives
#
# By the end of this tutorial, you will be able to:
# - Use the Gemini API to generate content
# - Apply various prompt engineering techniques (zero-shot, few-shot, chain-of-thought)
# - Control generation parameters (temperature, top-p, max_output_tokens)
# - Generate structured outputs using schemas
# - Generate, execute, and explain code
#
# ## Prerequisites
#
# Before starting, make sure you have:
# - Obtained a Gemini API key from [AI Studio](https://aistudio.google.com/app/api-keys). Follow instructions [here](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb)
# - Installed the required dependencies listed in `pyproject.toml` via `uv sync`

# %% [markdown]
# ## Part 1: Getting Started with the Gemini API

# %% [markdown]
# ### Import the SDK and Helpers

# %%
from google import genai
from google.genai import types
from IPython.display import HTML, Markdown, display
from google.api_core import retry

# %% [markdown]
# ### Set Up Retry Helper
#
# This allows you to run all cells without worrying about per-minute quota limits.
# The retry helper will automatically retry requests that fail due to rate limiting (429) or service unavailability (503).

# %%
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)

# %% [markdown]
# ### Initialize the Client
#
# The Gemini API uses a `Client` object to make requests.
# The client handles authentication and lets you control which backend to use (Gemini API or Vertex AI).

# %%
import google.colab.userdata

api_key = google.colab.userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

# %% [markdown]
# **Note:** Use the below code if you decide to run your code locally. We highly recommend using Google Colab

# %% [markdown]
# ### Run Your First Prompt
#
# Let's start with a simple text generation request. The `gemini-2.5-flash` model is a fast and efficient model suitable for most tasks.

# %%
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain AI to me like I'm a kid.")

print(response.text)

# %% [markdown]
# The response often comes back in markdown format, which you can render directly in notebooks:

# %%
Markdown(response.text)

# %% [markdown]
# ### Start a Chat
#
# The previous example uses a single-turn, text-in/text-out structure. You can also set up a multi-turn chat where the conversation state persists.

# %%
chat = client.chats.create(model='gemini-2.5-flash', history=[])
response = chat.send_message('Hello! My name is Alex.')
print(response.text)

# %%
response = chat.send_message('What is my name?')
print(response.text)

# %%
# The chat object maintains conversation state
response = chat.send_message('Do you remember what my name is?')
print(response.text)

# %% [markdown]
# ### Choose a Model
#
# The Gemini API provides access to multiple models from the Gemini family.
# Each model has different capabilities, token limits, and performance characteristics.
# You can list all available models:

# %%
for model in client.models.list():
    print(model.name)

# %% [markdown]
# The `models.list()` response also returns additional information about each model's capabilities, like token limits and supported parameters:

# %%
from pprint import pprint

for model in client.models.list():
    if model.name == 'models/gemini-2.5-flash':
        pprint(model.to_json_dict())
        break

# %% [markdown]
# For more information about available models and their capabilities,
# see the [Gemini API model overview](https://ai.google.dev/gemini-api/docs/models/gemini).

# %% [markdown]
# ## Part 2: Generation Parameters
#
# Generation parameters allow you to control how the model generates text. Understanding these parameters is crucial for getting the best results for your specific use case.

# %% [markdown]
# ### Output Length
#
# When generating text with an LLM, the output length affects cost and performance. Generating more tokens increases computation, leading to higher energy consumption, latency, and cost.
#
# To stop the model from generating tokens past a limit, you can specify the `max_output_tokens` parameter. This parameter stops generation once the specified length is reached, but it doesn't influence the style or content of the output. You may need to adjust your prompt to get a complete response within the limit.

# %%
short_config = types.GenerateContentConfig(max_output_tokens=200)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=short_config,
    contents='Write a 1000 word essay on the importance of olives in modern society.')

print(response.text)
print(f"\nLength: {len(response.text)} characters")

# %%
# With a more appropriate prompt for the token limit
response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=short_config,
    contents='Write a short poem on the importance of olives in modern society.')

print(response.text)

# %% [markdown]
# ### Temperature
#
# Temperature controls the degree of randomness in token selection. Higher temperatures result in more diverse and creative outputs, while lower temperatures produce more deterministic and focused results.
#
# - **High temperature (1.0-2.0)**: More creative, diverse outputs
# - **Low temperature (0.0-0.5)**: More deterministic, focused outputs
# - **Temperature 0.0**: Greedy decoding (selects the most probable token at each step)
#
# Temperature doesn't provide guarantees of randomness, but it can be used to "nudge" the output in the desired direction.

# %%
high_temp_config = types.GenerateContentConfig(temperature=2.0)

for _ in range(5):
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        config=high_temp_config,
        contents='Pick a random colour... (respond in a single word)')
    
    if response.text:
        print(response.text, '-' * 25)

# %%
# Now try with low temperature
low_temp_config = types.GenerateContentConfig(temperature=0.0)

for _ in range(5):
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        config=low_temp_config,
        contents='Pick a random colour... (respond in a single word)')
    
    if response.text:
        print(response.text, '-' * 25)

# %% [markdown]
# ### Top-P
#
# Like temperature, the top-P parameter is also used to control the diversity of the model's output.
#
# Top-P defines the probability threshold that, once cumulatively exceeded, tokens stop being selected as candidates.
# A top-P of 0 is typically equivalent to greedy decoding, and a top-P of 1 typically selects from every token in the model's vocabulary.
#
# **Note:** Top-K is not configurable in the Gemini 2.5 series of models, but can be changed in older models.
# Top-K is a positive integer that defines the number of most probable tokens from which to select the output token.

# %%
model_config = types.GenerateContentConfig(
    # These are the default values for gemini-2.5-flash
    temperature=1.0,
    top_p=0.95,
)

story_prompt = "You are a creative writer. Write a short story about a cat who goes on an adventure."
response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=model_config,
    contents=story_prompt)

print(response.text)

# %% [markdown]
# ## Part 3: Prompt Engineering Techniques
#
# Prompt engineering is the practice of designing effective prompts to get the best results from language models.
# This section covers several key techniques based on the [Gemini API prompting strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies).

# %% [markdown]
# ### Zero-Shot Prompting
#
# Zero-shot prompts describe the request directly without providing examples.
# The model relies on its training to understand and complete the task.
#
# Zero-shot prompting works well for:
# - Simple classification tasks
# - Well-defined tasks the model was trained on
# - When you want to avoid providing examples

# %%
model_config = types.GenerateContentConfig(
    temperature=0.1,
    top_p=1,
    max_output_tokens=5,
)

zero_shot_prompt = """Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE.
Review: "Her" is a disturbing study revealing the direction
humanity is headed if AI is allowed to keep evolving,
unchecked. I wish there were more movies like this masterpiece.
Sentiment: """

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=model_config,
    contents=zero_shot_prompt)

print(response.text)

# %% [markdown]
# #### Enum Mode
#
# Sometimes models can produce more text than you want, or include explanatory text. The Gemini API has an **Enum mode** feature that allows you to constrain the output to a fixed set of values. This ensures you get exactly one of the specified options.

# %%
import enum

class Sentiment(enum.Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        response_mime_type="text/x.enum",
        response_schema=Sentiment
    ),
    contents=zero_shot_prompt)

print(response.text)

# %% [markdown]
# When using constrained output like an enum, the Python SDK will attempt to convert the model's text response into a Python object automatically. It's stored in the `response.parsed` field:

# %%
enum_response = response.parsed
print(enum_response)
print(type(enum_response))

# %% [markdown]
# ### Few-Shot Prompting
#
# Providing examples of the expected response is known as "few-shot" prompting. When you provide one example, it's "one-shot"; multiple examples make it "few-shot."
#
# Few-shot prompting works well for:
# - Tasks with specific output formats
# - When you want to demonstrate the desired style or structure
# - Complex tasks that benefit from examples

# %%
few_shot_prompt = """Parse a customer's pizza order into valid JSON:

EXAMPLE:
I want a small pizza with cheese, tomato sauce, and pepperoni.
JSON Response:
```
{
"size": "small",
"type": "normal",
"ingredients": ["cheese", "tomato sauce", "pepperoni"]
}
```

EXAMPLE:
Can I get a large pizza with tomato sauce, basil and mozzarella
JSON Response:
```
{
"size": "large",
"type": "normal",
"ingredients": ["tomato sauce", "basil", "mozzarella"]
}
```

ORDER:
"""

customer_order = "Give me a large with cheese & pineapple"

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=1,
        max_output_tokens=250,
    ),
    contents=[few_shot_prompt, customer_order])

print(response.text)

# %% [markdown]
# #### JSON Mode
#
# To ensure you only receive JSON (with no other text or markdown), and to provide control over the schema, you can use the Gemini API's **JSON mode**. This forces the model to constrain decoding according to the supplied schema.

# %%
import typing_extensions as typing

class PizzaOrder(typing.TypedDict):
    size: str
    ingredients: list[str]
    type: str

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=PizzaOrder,
    ),
    contents="Can I have a large dessert pizza with apple and chocolate")

print(response.text)

# %% [markdown]
# ### Chain of Thought (CoT) Prompting
#
# Direct prompting can return answers quickly, but they can be prone to errors, especially for reasoning tasks. Chain-of-Thought prompting instructs the model to output intermediate reasoning steps, which typically leads to better results, especially when combined with few-shot examples.
#
# **Note:** This technique doesn't completely eliminate errors, and it tends to cost more due to increased token usage. However, it's very effective for complex reasoning tasks.

# %%
prompt = """When I was 4 years old, my partner was 3 times my age. Now, I
am 20 years old. How old is my partner? Return the answer directly."""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt)

print(response.text)

# %% [markdown]
# Now try the same problem with chain-of-thought prompting:

# %%
prompt = """When I was 4 years old, my partner was 3 times my age. Now,
I am 20 years old. How old is my partner? Let's think step by step."""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt)

Markdown(response.text)

# %% [markdown]
# ### System Instructions
#
# System instructions allow you to set the behavior, tone, and role of the model for the entire conversation. This is more efficient than including instructions in every user message.
#
# System instructions are useful for:
# - Setting the model's role (e.g., "You are a helpful assistant")
# - Defining output format preferences
# - Establishing guidelines that apply to all interactions

# %%
config = types.GenerateContentConfig(
    system_instruction="You are a helpful coding assistant. Always provide code examples with clear explanations. Use Python 3.10+ syntax."
)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=config,
    contents="How do I read a CSV file in Python?")

Markdown(response.text)

# %% [markdown]
# ### Thinking Mode
#
# The experimental Gemini Flash 2.5 "Thinking" model has been trained to generate the "thinking process" the model goes through as part of its response. This model is capable of stronger reasoning capabilities.
#
# Using a thinking mode model can provide high-quality responses without needing specialized prompting techniques. The model generates relevant intermediate thoughts that are then used as context for the final response.
#
# **Note:** When using the API, you get the final response, but the intermediate thoughts are not captured. To see the thinking process, you can try the thinking mode model in [AI Studio](https://aistudio.google.com/prompts/new_chat?model=gemini-2.5-flash-thinking-exp-01-21).

# %%
import io

response = client.models.generate_content_stream(
    model='gemini-2.5-flash-thinking-exp',
    contents='Who was the youngest author listed on the transformers NLP paper?',
)

buf = io.StringIO()
for chunk in response:
    buf.write(chunk.text)
    # Display the response as it is streamed
    print(chunk.text, end='')

# And then render the finished response as formatted markdown
from IPython.display import clear_output
clear_output()
Markdown(buf.getvalue())

# %% [markdown]
# ## Part 4: Code Generation and Execution
#
# The Gemini family of models can generate code, configuration files, and scripts. This is helpful when learning to code, learning a new language, or rapidly generating a first draft.
#
# **Important:** Since LLMs can make mistakes and may repeat training data, it's essential to read and test your code first, and comply with any relevant licenses.

# %% [markdown]
# ### Generating Code

# %%
code_prompt = """
Write a Python function to calculate the factorial of a number. No explanation, provide only the code.
"""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        temperature=1,
        top_p=1,
        max_output_tokens=1024,
    ),
    contents=code_prompt)

Markdown(response.text)

# %% [markdown]
# ### Code Execution
#
# The Gemini API can automatically run generated code and return the output. This is useful for mathematical calculations, data processing, and other computational tasks.

# %%
from pprint import pprint

config = types.GenerateContentConfig(
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
)

code_exec_prompt = """
Generate the first 14 odd prime numbers, then calculate their sum.
"""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=config,
    contents=code_exec_prompt)

for part in response.candidates[0].content.parts:
    pprint(part.to_json_dict())
    print("-----")

# %% [markdown]
# This response contains multiple parts:
# - Opening and closing text parts (regular responses)
# - `executable_code` part (generated code)
# - `code_execution_result` part (results from running the code)

# %%
for part in response.candidates[0].content.parts:
    if part.text:
        display(Markdown(part.text))
    elif part.executable_code:
        display(Markdown(f'```python\n{part.executable_code.code}\n```'))
    elif part.code_execution_result:
        if part.code_execution_result.outcome != 'OUTCOME_OK':
            display(Markdown(f'## Status {part.code_execution_result.outcome}'))
        
        display(Markdown(f'```\n{part.code_execution_result.output}\n```'))

# %% [markdown]
# ### Explaining Code
#
# The Gemini models can also explain code to you. This is useful for understanding unfamiliar codebases or learning new programming concepts.

# %%
# Example: Explain a simple Python function
code_to_explain = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

explain_prompt = f"""
Please explain what this function does, how it works, and what its time complexity is.

```python
{code_to_explain}
```
"""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=explain_prompt)

Markdown(response.text)

# %% [markdown]
# ## Summary
#
# In this tutorial, you've learned:
#
# 1. **Getting Started**: How to initialize the Gemini API client and make basic requests
# 2. **Generation Parameters**: How to control output length, temperature, and top-p
# 3. **Prompt Engineering**: Zero-shot, few-shot, chain-of-thought, and system instructions
# 4. **Structured Output**: Using enums and JSON schemas to constrain model outputs
# 5. **Code Generation**: Generating, executing, and explaining code
#
# ## Next Steps
#
# - Continue to `day-1b-evaluation-and-structured-output.py` to learn about evaluation methods
# - Explore the [Gemini API documentation](https://ai.google.dev/gemini-api/docs) for more advanced features
# - Review the [prompting strategies guide](https://ai.google.dev/gemini-api/docs/prompting-strategies) for more techniques
# - Check out the [Gemini API cookbook](https://github.com/google-gemini/cookbook) for more examples
# - Try building your own application using the techniques you've learned

# %% [markdown]
# ## References
#
# - [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
# - [Gemini API Prompting Strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)
# - [Gemini API Models Overview](https://ai.google.dev/gemini-api/docs/models/gemini)
# - [Gemini API Cookbook](https://github.com/google-gemini/cookbook)


