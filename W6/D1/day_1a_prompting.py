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
# # Day 1a - Prompting with OpenRouter AI
#
# This tutorial introduces you to the fundamentals of working with OpenRouter AI, including prompt engineering techniques and code generation.
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
# - Obtained an OpenRouter API key from [OpenRouter](https://openrouter.ai/keys)
# - Installed the required dependencies listed in `pyproject.toml` via `uv sync`

# %% [markdown]
# ## Part 1: Getting Started with OpenRouter AI

# %% [markdown]
# ### Import the SDK and Helpers

# %%
from openai import OpenAI
from IPython.display import Markdown, display
import os

# %% [markdown]
# ### Initialize the Client
#
# OpenRouter provides a unified API that gives you access to hundreds of AI models through a single endpoint.
# We use the OpenAI SDK with OpenRouter's base URL for compatibility.

# %%
# Get API key from environment variable or use the commented line for Colab
# For Colab: import google.colab.userdata; api_key = google.colab.userdata.get('OPENROUTER_API_KEY')
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')
if not api_key:
    raise ValueError("Please set OPENROUTER_API_KEY environment variable. Get your key from https://openrouter.ai/keys")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# %% [markdown]
# ### Choose a Model
#
# OpenRouter provides access to hundreds of AI models from various providers.
# Each model has different capabilities, token limits, and performance characteristics.
# You can list all available models:

# %%
# List available models from OpenRouter
import requests

models_response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)
models_data = models_response.json()

# Print model IDs
for model in models_data.get("data", [])[:20]:  # Show first 20 models
    print(f"{model.get('id')} - {model.get('name', 'N/A')}")

# %%
# Default model to use
# You can use any model available on OpenRouter.
# Check https://openrouter.ai/models for available models.
DEFAULT_MODEL = "deepseek/deepseek-chat"

# %% [markdown]
# ### Run Your First Prompt
#
# Let's start with a simple text generation request. We'll use a fast and efficient model suitable for most tasks.

# %%
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is OpenRouter AI?"}
    ]
)

print(response.choices[0].message.content)

# %% [markdown]
# The response often comes back in markdown format, which you can render directly in notebooks:

# %%
Markdown(response.choices[0].message.content)

# %% [markdown]
# ### Start a Chat
#
# The previous example uses a single-turn, text-in/text-out structure. You can also set up a multi-turn chat where the conversation state persists.

# %%
# For chat, we maintain conversation history manually
messages = []
messages.append({"role": "user", "content": "Hello! My name is Adam. And I am 11 years old."})
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=200,
    messages=messages
)
messages.append({"role": "assistant", "content": response.choices[0].message.content})
print(response.choices[0].message.content)

# %%
messages.append({"role": "user", "content": "Are you happy?"})
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=200,
    messages=messages
)
messages.append({"role": "assistant", "content": response.choices[0].message.content})
print(response.choices[0].message.content)

# %%
# The messages list maintains conversation state
messages.append({"role": "user", "content": "Who am I?"})
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=200,
    messages=messages
)
messages.append({"role": "assistant", "content": response.choices[0].message.content})
print(response.choices[0].message.content)

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
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "user", "content": "Write a 1000 word essay on the importance of olives in modern society."}
    ],
    max_tokens=25
)

print(response.choices[0].message.content)
print(f"\nLength: {len(response.choices[0].message.content)} characters")

# %%
# With a more appropriate prompt for the token limit
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "user", "content": "Write a short poem on the importance of olives in modern society."}
    ],
    max_tokens=25
)

print(response.choices[0].message.content)

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
for _ in range(3):
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": "Pick a random colour... (respond in a single word)"}
        ],
        temperature=2.0
    )
    
    if response.choices[0].message.content:
        print(response.choices[0].message.content, '-' * 25)

# %%
# Now try with low temperature
for _ in range(3):
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": "Pick a random colour... (respond in a single word)"}
        ],
        temperature=0.0
    )
    
    if response.choices[0].message.content:
        print(response.choices[0].message.content, '-' * 25)

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
story_prompt = "You are a creative writer. Write a short story about a cat who goes on an adventure."
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "user", "content": story_prompt}
    ],
    temperature=1.0,
    top_p=0.95
)

print(response.choices[0].message.content)

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
zero_shot_prompt = (
    "Classify restaurant reviews as POSITIVE, NEUTRAL or NEGATIVE."
    "\nReview: 'The food was great and the service was excellent. Delightful food and service.'",
    "\nSentiment: "
)

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "user", "content": zero_shot_prompt}
    ],
    temperature=0.1,
    top_p=1,
    max_tokens=5
)

print(response.choices[0].message.content)

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

# For structured output, we use JSON mode with a schema description
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {
            "role": "user",
            "content": f"""{zero_shot_prompt}
Respond with only one word: POSITIVE, NEUTRAL, or NEGATIVE."""
        }
    ],
    temperature=0.1,
    max_tokens=10
)

print(response.choices[0].message.content)

# %% [markdown]
# When using constrained output like an enum, the Python SDK will attempt to convert the model's text response into a Python object automatically. It's stored in the `response.parsed` field:

# %%
# Parse the response to get the enum value
response_text = response.choices[0].message.content.strip().upper()
try:
    enum_response = Sentiment[response_text]
except (KeyError, ValueError):
    # Fallback: try to match the value
    for sentiment in Sentiment:
        if sentiment.value.upper() in response_text or response_text in sentiment.value.upper():
            enum_response = sentiment
            break
    else:
        enum_response = None

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
few_shot_prompt = (
    "apple -> a.p.p.l.e"
    "\nbanana -> b.a.n.a.n.a"
    "\ncherry -> c.h.e.r.r.y"
)

user_input = "berry -> "

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "user", "content": few_shot_prompt + "\n" + user_input}
    ],
    temperature=0.1,
    top_p=1,
    max_tokens=250
)

print(response.choices[0].message.content)

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

# Use JSON mode for structured output
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=250,
    messages=[
        {
            "role": "user",
            "content": """Parse this pizza order into JSON with fields: size, ingredients (array), and type.
Order: Can I have a large dessert pizza with apple and chocolate
Respond with only valid JSON, no other text."""
        }
    ],
    temperature=0.1,
    response_format={"type": "json_object"}
)

print(response.choices[0].message.content)

# %% [markdown]
# ### Chain of Thought (CoT) Prompting
#
# Direct prompting can return answers quickly, but they can be prone to errors, especially for reasoning tasks. Chain-of-Thought prompting instructs the model to output intermediate reasoning steps, which typically leads to better results, especially when combined with few-shot examples.
#
# **Note:** This technique doesn't completely eliminate errors, and it tends to cost more due to increased token usage. However, it's very effective for complex reasoning tasks.

# %%
prompt = """When I was 4 years old, my partner was 3 times my age. Now, I
am 20 years old. How old is my partner? Return the answer directly."""

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=250,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

# %% [markdown]
# Now try the same problem with chain-of-thought prompting:

# %%
prompt = (
    "When I was 4 years old, my partner was 3 times my age. Now,"
    "I am 20 years old. How old is my partner?"
)

cot_trigger = "Let's think step by step."

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=250,
    messages=[
        {"role": "user", "content": prompt + " " + cot_trigger}
    ]
)

Markdown(response.choices[0].message.content)

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
system_prompt = (
    "You are a helpful coding assistant."
    " Always provide code examples with clear explanations."
    " Use Python 3.10+ syntax."
)
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=250,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "How do I read a CSV file in Python?"
        }
    ]
)

Markdown(response.choices[0].message.content)

# %% [markdown]
# ### Thinking Mode
#
# Some models on OpenRouter support "thinking" or reasoning modes that generate intermediate reasoning steps.
# These models can provide high-quality responses without needing specialized prompting techniques.
#
# **Note:** Check [OpenRouter Models](https://openrouter.ai/models) for models with reasoning capabilities.

# %%
import io

# Use streaming for real-time response
stream = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Who was the youngest author listed on the transformers NLP paper?"}
    ],
    stream=True
)

buf = io.StringIO()
for chunk in stream:
    if chunk.choices[0].delta.content:
        content = chunk.choices[0].delta.content
        buf.write(content)
        # Display the response as it is streamed
        print(content, end='', flush=True)

# And then render the finished response as formatted markdown
from IPython.display import clear_output
clear_output()
Markdown(buf.getvalue())

# %% [markdown]
# ## Part 4: Code Generation
#
# The Gemini family of models can generate code, configuration files, and scripts. This is helpful when learning to code, learning a new language, or rapidly generating a first draft.
#
# **Important:** Since LLMs can make mistakes and may repeat training data, it's essential to read and test your code first, and comply with any relevant licenses.

# %% [markdown]
# ### Generating Code

# %%
code_prompt = (
    "Write a Python function to calculate the factorial of a number."
    " No explanation, provide only the code."
)

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
        {"role": "user", "content": code_prompt}
    ],
    temperature=1,
    top_p=1,
    max_tokens=1024
)

Markdown(response.choices[0].message.content)

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

response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    max_tokens=1024,
    messages=[
        {"role": "user", "content": explain_prompt}
    ]
)

Markdown(response.choices[0].message.content)

# %% [markdown]
# ## Summary
#
# In this tutorial, you've learned:
#
# 1. **Getting Started**: How to initialize the OpenRouter API client and make basic requests
# 2. **Generation Parameters**: How to control output length, temperature, and top-p
# 3. **Prompt Engineering**: Zero-shot, few-shot, chain-of-thought, and system instructions
# 4. **Structured Output**: Using enums and JSON schemas to constrain model outputs
# 5. **Code Generation**: Generating and explaining code
