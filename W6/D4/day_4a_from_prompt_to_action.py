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
#     display_name: Python 3
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
# - ✅ Configure your API key to use the Gemini model
# - ✅ Build your first simple agent
# - ✅ Run your agent and watch it use a tool (like Google Search) to answer a question
#
# ## ‼️ Please Read
#
# > ❌ **ℹ️ Note: No submission required!**
# > This notebook is for your hands-on practice and learning only. You **do not** need to submit it anywhere to complete the course.
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
# pip install google-adk
# ```

# %% [markdown]
# ### 1.2: Configure your Gemini API Key
#
# This notebook uses the [Gemini API](https://ai.google.dev/gemini-api/docs), which requires authentication.
#
# **1. Get your API key**
#
# If you don't have one already, create an [API key in Google AI Studio](https://aistudio.google.com/app/api-keys).
#
# **2. Set your API key as an environment variable**
#
# Set the `GOOGLE_API_KEY` environment variable with your API key. You can do this by:
#
# - Setting it in your shell: `export GOOGLE_API_KEY="your-api-key-here"`
# - Or setting it in your Python code (see the cell below)
#
# **3. Authenticate in the notebook**
#
# Run the cell below to complete authentication.

# %%
import os

# Set your API key here, or set it as an environment variable before running
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # If not set as environment variable, you can set it directly here (not recommended for production)
    # GOOGLE_API_KEY = "your-api-key-here"
    print("⚠️ Warning: GOOGLE_API_KEY not found. Please set it as an environment variable or in the code above.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")

# %% [markdown]
# ### 1.3: Import ADK components
#
# Now, import the specific components you'll need from the Agent Development Kit and the Generative AI library. This keeps your code organized and ensures we have access to the necessary building blocks.

# %%
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("✅ ADK components imported successfully.")

# %% [markdown]
# ### 1.4: Helper functions
#
# We'll define some helper functions for accessing the ADK web UI.

# %%
# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Helper function to get the ADK web UI URL
# Note: This is a simplified version. In production, you may need to configure
# proxy settings based on your deployment environment.
def get_adk_proxy_url():
    """
    Get the URL prefix for accessing the ADK web UI.
    Adjust this based on your deployment environment.
    """
    ADK_PORT = "8000"
    # For local development, the UI is typically available at http://localhost:8000
    # For cloud environments, you may need to configure proxy settings
    url_prefix = f"/proxy/{ADK_PORT}"
    return url_prefix


print("✅ Helper functions defined.")

# %% [markdown]
# ### 1.5: Configure Retry Options
#
# When working with LLMs, you may encounter transient errors like rate limits or temporary service unavailability. Retry options automatically handle these failures by retrying the request with exponential backoff.

# %%
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

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
# In this notebook, we'll build an agent that can take the action of searching Google. Let's see the difference!

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
# - **model**: The specific LLM that will power the agent's reasoning. We'll use "gemini-2.5-flash-lite".
# - **instruction**: The agent's guiding prompt. This tells the agent what its goal is and how to behave.
# - **tools**: A list of [tools](https://google.github.io/adk-docs/tools/) that the agent can use. To start, we'll give it the `google_search` tool, which lets it find up-to-date information online.

# %%
root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

print("✅ Root Agent defined.")

# %% [markdown]
# ### 2.3 Run your agent
#
# Now it's time to bring your agent to life and send it a query. To do this, you need a [`Runner`](https://google.github.io/adk-docs/runtime/), which is the central component within ADK that acts as the orchestrator. It manages the conversation, sends our messages to the agent, and handles its responses.
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
    "What is Agent Development Kit from Google? What languages is the SDK available in?"
)

# %% [markdown]
# You can see a summary of ADK and its available languages in the response.
#
# ### 2.4 How does it work?
#
# The agent performed a Google Search to get the latest information about ADK, and it knew to use this tool because:
#
# 1. The agent inspects and is aware of which tools it has available to use.
# 2. The agent's instructions specify the use of the search tool to get current information or if it is unsure of an answer.
#
# The best way to see the full, detailed trace of the agent's thoughts and actions is in the **ADK web UI**, which we'll set up later in this notebook.
#
# And we'll cover more detailed workflows for logging and observability later in the course.

# %% [markdown]
# ### 🚀 2.5 Your Turn!
#
# This is your chance to see the agent in action. Ask it a question that requires current information.
#
# Try one of these, or make up your own:
#
# - What's the weather in London?
# - Who won the last soccer world cup?
# - What new movies are showing in theaters now?

# %%
response = await runner.run_debug("What's the weather in London?")

# %% [markdown]
# ---
#
# ## 💻 Section 3: Try the ADK Web Interface
#
# ### Overview
#
# ADK includes a built-in web interface for interactively chatting with, testing, and debugging your agents.
#
# <img width="1200" src="https://storage.googleapis.com/github-repo/adk-tutorials/day1/adk-web-ui.gif" alt="ADK Web UI" />
#
# To use the ADK web UI, you'll need to create an agent with Python files using the `adk create` command.
#
# Run the command below to generate a `sample-agent` folder that contains all the necessary files, including `agent.py` for your code, an `.env` file with your API key pre-configured, and an `__init__.py` file:

# %%
# !adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY

# %% [markdown]
# Get the URL prefix to access the ADK web UI:

# %%
url_prefix = get_adk_proxy_url()

# %% [markdown]
# Now we can run ADK web:

# %%
# !adk web --url_prefix {url_prefix}

# %% [markdown]
# Now you can access the ADK dev UI. The web interface allows you to interact with your ADK agent.
#
# Note: This sample agent does not have any tools enabled (like Google Search). It is a basic agent designed specifically to let you explore the UI features.

# %% [markdown]
# ---
#
# ## ✅ Congratulations!
#
# You've built and run your first agent with ADK! You've just seen the core concept of agent development in action.
#
# The big takeaway is that your agent didn't just *respond*—it **reasoned** that it needed more information and then **acted** by using a tool. This ability to take action is the foundation of all agent-based AI.
#
# **ℹ️ Note: No submission required!**
#
# This notebook is for your hands-on practice and learning only. You **do not** need to submit it anywhere to complete the course.
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
