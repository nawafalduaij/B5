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
# # Day 2 - Document Q&A with RAG using Chroma and the Gemini API
#
# Welcome to the Generative AI Course!
#
# Two big limitations of LLMs are 1) that they only "know" the information that they were trained on, and 2) that they have limited input context windows. A way to address both of these limitations is to use a technique called Retrieval Augmented Generation, or RAG.
#
# In this notebook you will use the Gemini API to create a vector database, retrieve answers to questions from the database and generate a final answer. You will use [Chroma](https://docs.trychroma.com/), an open-source vector database.
#
# **Prerequisites**:
# - You need a Google Cloud Project with the Gemini API enabled.
# - You need an API key stored in the `GOOGLE_API_KEY` environment variable.

# %% [markdown]
# ## Setup
#
# ```bash
# pip install -U -q "google-genai" "chromadb"
# ```

# %%
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from google.api_core import retry
from IPython.display import Markdown

# %% [markdown]
# ### Set up your API key

# %%
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# %% [markdown]
# ### Explore available models
# `text-embedding-004` is the most recent generally-available embedding model, so you will use it for this exercise.

# %%
# Explore models that support embeddings
for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)

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
# Create a [custom function](https://docs.trychroma.com/guides/embeddings#custom-embedding-functions) to generate embeddings with the Gemini API. In this task, you are implementing a retrieval system, so the `task_type` for generating the *document* embeddings is `retrieval_document`. Later, you will use `retrieval_query` for the *query* embeddings.

# %%
# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]

# %% [markdown]
# Now create a [Chroma database client](https://docs.trychroma.com/getting-started) that uses the `GeminiEmbeddingFunction` and populate the database with the documents you defined above.

# %%
DB_NAME = "googlecardb"

embed_fn = GeminiEmbeddingFunction()
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
# Now that you have found a relevant passage from the set of documents (the *retrieval* step), you can now assemble a generation prompt to have the Gemini API *generate* a final answer.

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
# Now use the `generate_content` method to to generate an answer to the question.

# %%
answer = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\n--- Model Answer ---")
Markdown(answer.text)

# %% [markdown]
# ## Next steps
#
# Congrats on building a Retrieval-Augmented Generation app!
#
# To learn more about using embeddings in the Gemini API, check out the [Intro to embeddings](https://ai.google.dev/gemini-api/docs/embeddings) or to learn more fundamentals, study the [embeddings chapter](https://developers.google.com/machine-learning/crash-course/embeddings) of the Machine Learning Crash Course.
#
# For a hosted RAG system, check out the [Semantic Retrieval service](https://ai.google.dev/gemini-api/docs/semantic_retrieval) in the Gemini API. You can implement question-answering on your own documents in a single request, or host a database for even faster responses.
