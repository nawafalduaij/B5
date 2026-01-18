# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Day 4 - Google Search Grounding with the Gemini API
#
# Welcome to the Generative AI Course!
#
# In this tutorial, you will use [Google Search](https://google.com/) results with the Gemini API in a technique called grounding. This connects the model to verifiable sources of information, allowing it to answer questions about current events or facts that are not in its training data.
#
# **Prerequisites**:
# - A Google Cloud Project with the Gemini API enabled.
# - An API key stored in the `GOOGLE_API_KEY` environment variable.
# - Note: Grounding with Google Search is a paid feature in some tiers/models, or available in the free tier for `gemini-2.0-flash`.

# %% [markdown]
# ## Setup
#
# ```bash
# pip install -U -q "google-genai"
# ```

# %%
import os
from google import genai
from google.genai import types
from IPython.display import Markdown, display

# %% [markdown]
# ### Set up your API key

# %%
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# %% [markdown]
# ## Standard Generation vs. Grounded Generation
#
# First, let's see how the model answers a question about a very recent event *without* grounding. It might hallucinate or say it doesn't know.

# %%
prompt = "Who won the most recent Euro football championship and what was the score?"

print(f"--- Prompt: {prompt} ---\n")

# Without grounding
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt
)
print("Without Grounding:")
display(Markdown(response.text))

# %% [markdown]
# ## Enable Google Search Grounding
#
# To enable grounding, we simply add the `google_search` tool to the configuration.

# %%
config_with_grounding = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    response_modalities=["TEXT"]
)

response_grounded = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt,
    config=config_with_grounding,
)

print("\nWith Grounding:")
display(Markdown(response_grounded.text))

# %% [markdown]
# ## Inspecting Grounding Metadata
#
# the Gemini API returns metadata about the search results used to ground the answer. This is crucial for verifying accuracy and providing citations to users.

# %%
grounding_metadata = response_grounded.candidates[0].grounding_metadata

if grounding_metadata:
    print("\n--- Grounding Metadata ---")
    
    # 1. Search Entry Point: A pre-rendered HTML snippet from Google (useful for web UIs)
    if grounding_metadata.search_entry_point:
        print("\n[Search Entry Point available using: grounding_metadata.search_entry_point.rendered_content]")
        # display(HTML(grounding_metadata.search_entry_point.rendered_content)) # Uncomment in a notebook environment

    # 2. Grounding Chunks: The specific web sources used
    if grounding_metadata.grounding_chunks:
        print("\nSources:")
        for i, chunk in enumerate(grounding_metadata.grounding_chunks):
            if chunk.web:
                print(f"{i+1}. {chunk.web.title} ({chunk.web.uri})")
else:
    print("No grounding metadata found (grounding might not have been triggered).")

# %% [markdown]
# ## Adding Inline Citations
#
# You can use the `grounding_supports` data to detect which parts of the text correspond to which sources. Here is a helper function to format the output with inline citations (e.g., `[1]`).

# %%
def print_with_citations(response):
    candidate = response.candidates[0]
    text = candidate.content.parts[0].text
    grounding_metadata = candidate.grounding_metadata
    
    if not grounding_metadata or not grounding_metadata.grounding_supports:
        print(text)
        return

    # 'grounding_supports' tells us which segment of text maps to which chunk index.
    # We do a simple print here. For a robust UI, you would insert markers into the string.
    
    print("Answer with Sources:\n")
    print(text)
    
    print("\n--- References ---")
    chunks = grounding_metadata.grounding_chunks
    for support in grounding_metadata.grounding_supports:
        # Each support points to a segment of text and a list of chunk indices
        indices = support.grounding_chunk_indices
        for idx in indices:
            chunk = chunks[idx]
            if chunk.web:
                 print(f"- {chunk.web.title}: {chunk.web.uri}")

# Run it on our previous response
print_with_citations(response_grounded)

# %% [markdown]
# ## Grounding with other tools
#
# You can mix Google Search with other tools (though some limitations apply depending on the model version).
# By creating a chat, you can let the model decide when to search.

# %%
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=config_with_grounding # Grounding enabled for the whole chat
)

response = chat.send_message("What is the stock price of Alphabet Inc (GOOG) right now?")
display(Markdown(response.text))

# Follow up with something that doesn't strictly need search, but the model might check anyway.
response = chat.send_message("Is that higher or lower than it was last week?")
display(Markdown(response.text))

# %% [markdown]
# ## Further reading
#
# When using search grounding, there are some specific requirements that you must follow, including when and how to show search suggestions, and how to use the grounding links.  Be sure to read and follow the details in the [search grounding capability guide](https://ai.google.dev/gemini-api/docs/grounding) and the [search suggestions guide](https://ai.google.dev/gemini-api/docs/grounding/search-suggestions).
#
# Also check out some more compelling examples of using search grounding with the Live API in the [cookbook](https://github.com/google-gemini/cookbook/), like [this example that uses Google Maps to plot Search results on a map](https://github.com/google-gemini/cookbook/blob/main/examples/LiveAPI_plotting_and_mapping.ipynb) in an audio conversation, or [this example](https://github.com/google-gemini/cookbook/blob/main/examples/Search_grounding_for_research_report.ipynb) that builds a comprehensive research report.
