# Instructions

## Lab Setup

### A. Local Setup

1. Use `git clone` to clone the repository
2. Use `uv sync` to install the dependencies in a virtual environment

> [How to run VS Code Notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

### B. Colab Setup

1. Open the notebook in Colab
2. Use `git clone` to clone the repository (to get helper files, assets, and datasets)
3. Run `%cd <path to the repository>`
4. Sync dependencies from `pyproject.toml` into the Colab system environment: `!uv sync --system` (notice the `!` prefix tells notebooks this is a shell command)

> Remember to save your work. Any files created in Colab will be lost if you don't save them elsewhere (Download).

#### How to Download Kaggle Dataset to Colab?

# 

### Prerequisite

First, obtain your API token from Kaggle:

1. Go to [**Kaggle Settings** ](https://www.kaggle.com/settings)
2. Scroll to the **API** section and click **Create New Token**
3. Copy that token and insert it as a Colab secret under with the label `KAGGLE_API_TOKEN`

```python
import os

# This reads your colab secrets
# and set the environment variables on them
from google.colab import userdata
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_API_TOKEN')
os.environ['KAGGLE_USERNAME'] = "KAGGLE_COLAB"

# Example Dataset
dataset_name = "shuyangli94/food-com-recipes-and-user-interactions"
!kaggle datasets download -d {dataset_name} 
```

### C. Code Locally, Run in Colab (Hybrid)

> See: [Connect notebooks to Colab servers](https://marketplace.visualstudio.com/items?itemName=google.colab)>