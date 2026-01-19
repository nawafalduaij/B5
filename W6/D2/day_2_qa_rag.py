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
# # Day 2 - Document Q&A with RAG using Chroma and OpenAI/OpenRouter API
#
# Welcome to the Generative AI Course!
#
# Two big limitations of LLMs are 1) that they only "know" the information that they were trained on, and 2) that they have limited input context windows. A way to address both of these limitations is to use a technique called Retrieval Augmented Generation, or RAG.
#
# In this notebook you will use the OpenAI API for embeddings and OpenRouter API for generation to create a vector database, retrieve answers to questions from the database and generate a final answer. You will use [Chroma](https://docs.trychroma.com/), an open-source vector database.
#
# **Prerequisites**:
# - You need an OpenAI API key stored in the `OPENAI_API_KEY` environment variable.
# - You need an OpenRouter API key stored in the `OPENROUTER_API_KEY` environment variable.

# %% [markdown]
# ## Setup
#
# ```bash
# pip install -U -q "openai" "chromadb"
# ```

# %%
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI
from IPython.display import Markdown
import time

# %% [markdown]
# ### Set up your API keys

# %%
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

if not OPENROUTER_API_KEY:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

# OpenAI client for embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# OpenRouter client for generation
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# %% [markdown]
# ### Data
# Here is a small set of documents you will use to create an embedding database.

# %%
DOCUMENT1 = "Operating the Climate Control System  Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."
DOCUMENT2 = 'Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the "Navigation" icon to get directions to your destination or touch the "Music" icon to play your favorite songs.'
DOCUMENT3 = "Shifting Gears Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

# %% [markdown]
# ## Creating the embedding database with ChromaDB
#
# Create a [custom function](https://docs.trychroma.com/guides/embeddings#custom-embedding-functions) to generate embeddings with the OpenAI API. OpenAI's text-embedding-3-small model works well for both document and query embeddings.

# %%
class OpenAIEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    # Note: OpenAI embeddings work well for both, but we keep this for consistency
    document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        # Retry logic for rate limits
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=input,
                )
                return [e.embedding for e in response.data]
            except Exception as e:
                if attempt < max_retries - 1 and (hasattr(e, 'status_code') and e.status_code in {429, 503}):
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise

# %% [markdown]
# Now create a [Chroma database client](https://docs.trychroma.com/getting-started) that uses the `OpenAIEmbeddingFunction` and populate the database with the documents you defined above.

# %%
DB_NAME = "googlecardb"

embed_fn = OpenAIEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

# Confirm insertion
print(f"Document count: {db.count()}")

# %% [markdown]
# ## Retrieval: Find relevant documents
#
# To search the Chroma database, call the `query` method. Note that you also switch to the `retrieval_query` mode of embedding generation.

# %%
# Switch to query mode when generating embeddings.
embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "How do you use the touchscreen to play music?"

result = db.query(query_texts=[query], n_results=1)
[all_passages] = result["documents"]

print("Retrieved Passage:")
print(all_passages[0])

# %% [markdown]
# ## Augmented generation: Answer the question
#
# Now that you have found a relevant passage from the set of documents (the *retrieval* step), you can now assemble a generation prompt to have the OpenRouter API *generate* a final answer.

# %%
query_oneline = query.replace("\n", " ")

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
"""

# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"

print("--- Generated Prompt ---")
print(prompt)

# %% [markdown]
# Now use the OpenRouter API to generate an answer to the question.

# %%
response = openrouter_client.chat.completions.create(
    model="openai/gpt-4o-mini",  # You can use any model available on OpenRouter
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
)

answer_text = response.choices[0].message.content

print("\n--- Model Answer ---")
Markdown(answer_text)

# %% [markdown]
# ## Next steps
#
# Congrats on building a Retrieval-Augmented Generation app!
#
# To learn more about using embeddings with OpenAI, check out the [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings).
#
# For more information about OpenRouter and available models, visit [OpenRouter Models](https://openrouter.ai/models).
