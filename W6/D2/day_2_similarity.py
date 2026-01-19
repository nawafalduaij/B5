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
# # Day 2 - Embeddings and Similarity Scores
#
# Welcome to the Generative AI Course!
#
# In this tutorial, you will use the OpenAI API's embedding endpoint to explore similarity scores.
#
# **Prerequisites**:
# - You need an OpenAI API key stored in the `OPENAI_API_KEY` environment variable.

# %% [markdown]
# ## Setup
# First, we'll install the necessary libraries.
#
# ```bash
# pip install -U -q "openai" pandas seaborn matplotlib
# ```

# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI

# %% [markdown]
# ### Set up your API key
# Ensure your `OPENAI_API_KEY` is set in your environment variables.

# %%
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

# %% [markdown]
# ## Calculate similarity scores
#
# This example embeds some variations on the pangram, `The quick brown fox jumps over the lazy dog`, including spelling mistakes and shortenings of the phrase. Another pangram and a somewhat unrelated phrase have been included for comparison.
#
# OpenAI embeddings work well for calculating similarity scores between texts.

# %%
texts = [
    'The quick brown fox jumps over the lazy dog.',
    'The quick rbown fox jumps over the lazy dog.',
    'teh fast fox jumps over the slow woofer.',
    'a quick brown fox jmps over lazy dog.',
    'brown fox jumping over dog',
    'fox > dog',
    # Alternative pangram for comparison:
    'The five boxing wizards jump quickly.',
    # Unrelated text, also for comparison:
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus et hendrerit massa. Sed pulvinar, nisi a lobortis sagittis, neque risus gravida dolor, in porta dui odio vel purus.',
]

response = client.embeddings.create(
    model='text-embedding-3-small',
    input=texts
)

# %% [markdown]
# Define a short helper function that will make it easier to display longer embedding texts in our visualisation.

# %%
def truncate(t: str, limit: int = 50) -> str:
    """Truncate labels to fit on the chart."""
    if len(t) > limit:
        return t[:limit-3] + '...'
    else:
        return t

truncated_texts = [truncate(t) for t in texts]

# %% [markdown]
# A similarity score of two embedding vectors can be obtained by calculating their inner product. If $\mathbf{u}$ is the first embedding vector, and $\\mathbf{v}$ the second, this is $\mathbf{u}^T \\mathbf{v}$. As the API provides embedding vectors that are normalised to unit length, this is also the cosine similarity.
#
# This score can be computed across all embeddings through the matrix self-multiplication: `df @ df.T`.
#
# Note that the range from 0.0 (completely dissimilar) to 1.0 (completely similar) is depicted in the heatmap from light (0.0) to dark (1.0).

# %%
# Set up the embeddings in a dataframe.
df = pd.DataFrame([e.embedding for e in response.data], index=truncated_texts)

# Perform the similarity calculation
sim = df @ df.T

# Draw!
plt.figure(figsize=(10, 8))
sns.heatmap(sim, vmin=0, vmax=1, cmap="Greens", annot=True)
plt.title("Semantic Similarity Heatmap")
plt.show()

# %% [markdown]
# You can see the scores for a particular term directly by looking it up in the dataframe.

# %%
print(sim['The quick brown fox jumps over the lazy dog.'].sort_values(ascending=False))

# %% [markdown]
# ## Further reading
#
# * Explore [OpenAI embeddings documentation](https://platform.openai.com/docs/guides/embeddings)
# * Learn more about [similarity search with embeddings](https://platform.openai.com/docs/guides/embeddings/use-cases)
