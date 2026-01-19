# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🚀 Your First AI Agent: From Prompt to Action
#
# This tutorial is your first step into building AI agents. An agent can do more than just respond to a prompt — it can **take actions** to find information or get things done.
#
# In this notebook, you'll:
#
# - ✅ Install [Agent Development Kit (ADK)](https://google.github.io/adk-docs/)
# - ✅ Build your first simple agent (we have configured it to use **OpenRouter** with the **DeepSeek** model via `LiteLlm`.)
# - ✅ Run your agent and watch it use a tool (like a calculator) to perform calculations
#
# **⏱️ Expected Reading Time:** 15 Minutes
#
# > ⏸️ **Note:** Avoid using the **Run all** cells command as this can trigger a QPM limit resulting in 429 errors when calling the backing model. Suggested flow is to run each cell in order - one at a time.

# %% [markdown]
# ## ⚙️ Section 1: Setup
#
# ### 1.1: Install dependencies
#
# To install and use ADK in your Python development environment, run:
#
# ```
# pip install google-adk litellm
# ```

# %% [markdown]
# ### 1.2: Configure your OpenRouter API Key
#
# This notebook uses **OpenRouter** to access the DeepSeek model.
#
# **1. Get your API key**
#
# Create an account and key at [openrouter.ai](https://openrouter.ai/).
#
# **2. Set your API key as an environment variable**
#
# Set the `OPENROUTER_API_KEY` environment variable.
#
# - Shell: `export OPENROUTER_API_KEY="sk-or-..."`
# - Python: (see cell below)
#
# **3. Authenticate in the notebook**
#
# Run the cell below to complete authentication.

# %%
import os

# Set your API key here, or set it as an environment variable before running
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("⚠️ Warning: OPENROUTER_API_KEY not found. Please set it as an environment variable.")
else:
    print("✅ Setup complete.")

# %% [markdown]
# ### 1.3: Import ADK components
#
# Now, import the specific components you'll need from the Agent Development Kit and the Generative AI library. This keeps your code organized and ensures we have access to the necessary building blocks.

# %%
from google.adk.agents import Agent
# Using LiteLlm to connect to DeepSeek via OpenRouter
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool

print("✅ ADK components imported successfully.")


# %% [markdown]
# ---
#
# ## 🤖 Section 2: Your first AI Agent with ADK
#
# ### 🤔 2.1 What is an AI Agent?
#
# You've probably used an LLM like Gemini before, where you give it a prompt and it gives you a text response.
#
# `Prompt -> LLM -> Text`
#
# An AI Agent takes this one step further. An agent can think, take actions, and observe the results of those actions to give you a better answer.
#
# `Prompt -> Agent -> Thought -> Action -> Observation -> Final Answer`
#
# In this notebook, we'll build an agent that can take the action of performing calculations. Let's see the difference!

# %% [markdown]
# ### 2.1.1: Create Calculator Tool
#
# We'll create a simple calculator tool that can perform basic math operations.

# %%
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression and return the result.
    
    This tool can perform basic arithmetic operations: addition (+), subtraction (-),
    multiplication (*), division (/), and exponentiation (**).
    
    Args:
        expression: A string containing a mathematical expression to evaluate.
                   Examples: "2 + 2", "10 * 5", "100 / 4", "2 ** 3"
    
    Returns:
        Dictionary with status and calculation result.
        Success: {"status": "success", "result": 4.0}
        Error: {"status": "error", "error_message": "Invalid expression"}
    """
    try:
        # Simple validation - only allow numbers, operators, and parentheses
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression.replace(" ", "")):
            return {
                "status": "error",
                "error_message": "Expression contains invalid characters. Only numbers and basic operators (+, -, *, /, **) are allowed."
            }
        
        # Evaluate the expression safely (restricted environment)
        result = eval(expression, {"__builtins__": {}}, {})
        
        return {
            "status": "success",
            "result": float(result) if isinstance(result, (int, float)) else result
        }
    except ZeroDivisionError:
        return {
            "status": "error",
            "error_message": "Division by zero is not allowed"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Calculation failed: {str(e)}"
        }


# Create the FunctionTool from our calculator function
calculator_tool = FunctionTool(func=calculate)

print("✅ Calculator tool created.")

# %% [markdown]
# ### 2.2 Define your agent
#
# Now, let's build our agent. We'll configure an `Agent` by setting its key properties, which tell it what to do and how to operate.
#
# To learn more, check out the documentation related to [agents in ADK](https://google.github.io/adk-docs/agents/).
#
# These are the main properties we'll set:
#
# - **name** and **description**: A simple name and description to identify our agent.
# - **model**: The specific LLM that will power the agent's reasoning. We'll use DeepSeek via OpenRouter.
# - **instruction**: The agent's guiding prompt. This tells the agent what its goal is and how to behave.
# - **tools**: A list of [tools](https://google.github.io/adk-docs/tools/) that the agent can use. We'll give it a simple calculator tool for performing math operations.

# %%
root_agent = Agent(
    name="calculator_assistant",
    # Using LiteLlm with DeepSeek model via OpenRouter
    model=LiteLlm(model="openrouter/deepseek/deepseek-chat"),
    description="A simple agent that can perform mathematical calculations.",
    instruction="You are a helpful calculator assistant. When users ask for calculations, use the calculate tool to perform the math operations.",
    tools=[calculator_tool],
)

print("✅ Root Agent defined.")

# %% [markdown]
# ### 2.3 Run your agent
#
# Now it's time to bring your agent to life and send it a query. To do this, you need a [`Runner`](https://google.github.io/adk-docs/runtime/), which is the central component within ADK that acts as the **orchestrator** - It manages the conversation:
#
# 1. sends our messages to the agent
# 2. and handles its responses
#
# **a. Create an `InMemoryRunner` and tell it to use our `root_agent`:**

# %%
runner = InMemoryRunner(agent=root_agent)

print("✅ Runner created.")

# %% [markdown]
# 👉 Note that we are using the Python Runner directly in this notebook. You can also run agents using ADK command-line tools such as `adk run`, `adk web`, or `adk api_server`. To learn more, check out the documentation related to [runtime in ADK](https://google.github.io/adk-docs/runtime/).

# %% [markdown]
# **b. Now you can call the `.run_debug()` method to send our prompt and get an answer.**
#
# 👉 This method abstracts the process of session creation and maintenance and is used in prototyping. We'll explore "what sessions are and how to create them" on Day 3.

# %%
response = await runner.run_debug(
    "What is 25 * 17 + 42?"
)

# %%
print(response[0].content.parts[-1].text)

# %% [markdown]
# You can see the calculation result in the response.
#
# ### 2.4 How does it work?
#
# The agent performed a calculation using the calculator tool, and it knew to use this tool because:
#
# 1. The agent inspects and is aware of which tools it has available to use.
# 2. The agent's instructions specify using the calculator tool for math operations.
#
# The best way to see the full, detailed trace of the agent's thoughts and actions is in the **ADK web UI**, which we'll set up later in this notebook.
#
# And we'll cover more detailed workflows for logging and observability later in the course.

# %% [markdown]
# ### 🚀 2.5 Your Turn!
#
# This is your chance to see the agent in action. Ask it to perform some calculations!
#
# Try one of these, or make up your own:
#
# - What is 123 * 456?
# - Calculate 1000 / 25 + 50
# - What's 2 to the power of 10?

# %%
response = await runner.run_debug("What is 123 * 456?")

# %% [markdown]
# ---
#
# ## ✅ Congratulations!
#
# You've built and run your first agent with ADK! You've just seen the core concept of agent development in action.
#
# The big takeaway is that your agent didn't just *respond*—it **reasoned** that it needed more information and then **acted** by using a tool. This ability to take action is the foundation of all agent-based AI.
#
# ### 📚 Learn More
#
# Refer to the following documentation to learn more:
#
# - [ADK Documentation](https://google.github.io/adk-docs/)
# - [ADK Quickstart for Python](https://google.github.io/adk-docs/get-started/python/)
# - [ADK Agents Overview](https://google.github.io/adk-docs/agents/)
# - [ADK Tools Overview](https://google.github.io/adk-docs/tools/)
#
# ### 🎯 Next Steps
#
# Ready for the next challenge? Continue to the next notebook to learn how to **architect multi-agent systems.**
