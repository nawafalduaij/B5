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
# # Day 2 - Classifying Embeddings with PyTorch and the OpenAI API
#
# ## Overview
#
# Welcome to the Generative AI Course. In this notebook, you'll learn to use the embeddings produced by the OpenAI API to train a model that can classify newsgroup posts into their categories (the newsgroup itself) from the post contents.
#
# This technique uses the OpenAI API's embeddings as input, avoiding the need to train on text input directly, and as a result it is able to perform quite well using relatively few examples.
#
# **Prerequisites**:
# - You need an OpenAI API key stored in the `OPENAI_API_KEY` environment variable.

# %% [markdown]
# ## Setup
#
# ```bash
# pip install -U -q "openai" scikit-learn torch tqdm pandas
# ```

# %%
import os
import re
import email
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from openai import OpenAI
import tqdm
import time

# %% [markdown]
# ### Set up your API key

# %%
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

# %% [markdown]
# ## Dataset
#
# The [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) contains 18,000 newsgroups posts on 20 topics divided into training and test sets.

# %%
newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# View list of class names for dataset
print(newsgroups_train.target_names)

# %% [markdown]
# Start by preprocessing the data. To remove any sensitive information like names and email addresses, you will take only the subject and body of each message.

# %%
def preprocess_newsgroup_row(data):
    # Extract only the subject and body
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    # Strip any remaining email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Truncate each entry to 5,000 characters
    text = text[:5000]
    return text

def preprocess_newsgroup_data(newsgroup_dataset):
    # Put data points into dataframe
    df = pd.DataFrame(
        {"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target}
    )
    # Clean up the text
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)
    # Match label to target name index
    df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])

    return df

# Apply preprocessing function to training and test datasets
df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

df_train.head()

# %% [markdown]
# Next, you will sample some of the data by taking 100 data points in the training dataset, and dropping a few of the categories to run through this tutorial. Choose the science categories to compare.

# %%
def sample_data(df, num_samples, classes_to_keep):
    # Sample rows, selecting num_samples of each Label.
    df = (
        df.groupby("Label")[df.columns]
        .apply(lambda x: x.sample(num_samples))
        .reset_index(drop=True)
    )

    df = df[df["Class Name"].str.contains(classes_to_keep)]

    # We have fewer categories now, so re-calibrate the label encoding.
    df["Class Name"] = df["Class Name"].astype("category")
    df["Encoded Label"] = df["Class Name"].cat.codes

    return df

TRAIN_NUM_SAMPLES = 100
TEST_NUM_SAMPLES = 25
CLASSES_TO_KEEP = "sci"

df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)

print(df_train.value_counts("Class Name"))

# %% [markdown]
# ## Create the embeddings
#
# In this section, you will generate embeddings for each piece of text using the OpenAI API embeddings endpoint.
#
# OpenAI's text-embedding-3-small model works well for classification tasks.

# %%
def embed_fn(text: str) -> list[float]:
    # Retry logic for rate limits
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1 and (hasattr(e, 'status_code') and e.status_code in {429, 503}):
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise

def create_embeddings(df):
    tqdm.tqdm.pandas()
    df["Embeddings"] = df["Text"].progress_apply(embed_fn)
    return df

print("Generating embeddings for Training set...")
df_train = create_embeddings(df_train)
print("Generating embeddings for Test set...")
df_test = create_embeddings(df_test)

df_train.head()

# %% [markdown]
# ## Build a classification model (PyTorch)
#
# Here you will define a simple model using PyTorch that accepts the embedding data as input.

# %%
# Prepare data for PyTorch
X_train = np.stack(df_train["Embeddings"].values)
y_train = df_train["Encoded Label"].values
X_test = np.stack(df_test["Embeddings"].values)
y_test = df_test["Encoded Label"].values

# Convert to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the Model
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, num_classes)
        # Note: CrossEntropyLoss includes Softmax, so we output raw logits

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

embedding_size = len(df_train["Embeddings"].iloc[0])
num_classes = len(df_train["Class Name"].unique())

model = Classifier(embedding_size, num_classes)
print(model)

# %% [markdown]
# ## Train the model

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print stats every 5 epochs or so
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# %% [markdown]
# ## Evaluate model performance

# %%
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# %% [markdown]
# ## Try a custom prediction

# %%
def make_prediction(text: str) -> torch.Tensor:
    """Infer categories from the provided text."""
    # Calculate embedding
    embedded_val = embed_fn(text)
    
    # Convert to tensor and add batch dimension
    inp = torch.tensor(np.array([embedded_val]), dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs = model(inp)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    return probs[0]

new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""

result = make_prediction(new_text)

# Get categories from the dataframe
categories = df_train["Class Name"].cat.categories

for idx, category in enumerate(categories):
    print(f"{category}: {result[idx] * 100:0.2f}%")

# %% [markdown]
# ## Further reading
#
# To explore training custom models with PyTorch further, check out the [PyTorch tutorials](https://pytorch.org/tutorials/).
