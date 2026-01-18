# Week 5 NLP Course

## Module 1: Text Analysis with Statistical NLP

**Module Overview**: This module teaches you how to work with text data using traditional statistical methods—the foundation that modern deep learning approaches build upon.

### Key Skills

- Build, evaluate, and improve supervised ML text classification pipelines
- Apply keyword-based information retrieval algorithms to find specific data in unstructured text
- Apply unsupervised ML to cluster documents and discover hidden topics in unorganized, unlabeled data

### Topics Covered

1. **NLP Introduction**: What is NLP, applications, challenges, NLP pipeline overview
2. **Corpus & EDA**: Exploratory data analysis on text corpora
3. **Text Preprocessing**: 
   - Cleaning (noise removal, URLs, HTML, punctuation)
   - Normalization (case, contractions, elongations, diacritics)
   - Basic tokenization (whitespace splitting)
   - Stemming
4. **Vectorization**:
   - Bag of Words (BoW)
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Sparse matrices
   - N-grams (unigrams, bigrams, trigrams)
5. **Text Classification**: 
   - Supervised ML with scikit-learn
   - Evaluation metrics
   - Building and improving classification pipelines
6. **Information Retrieval**: 
   - TF-IDF-based search
   - Cosine similarity
7. **Topic Modeling**: 
   - LDA (Latent Dirichlet Allocation)
   - Unsupervised document clustering

### Session Structure

- `01_intro.ipynb`: Introduction to NLP
- `02_corpus.ipynb`: Corpus exploration and EDA
- `03_prep.ipynb`: Text preprocessing
- `04_vectorization.ipynb`: Vectorization techniques
- `05_classifier_ex1.ipynb` & `06_classifier_ex2.ipynb`: Text classification exercises
- `07_information_retrieval_ex.ipynb`: Information retrieval
- `08_topic_modeling_ex.ipynb`: Topic modeling

### Stretch Exercises

- `regex.ipynb` & `regex_ex.ipynb`: Regular expressions for text processing
- `TF_TFIDF_From_Scratch.ipynb`: Implementing TF-IDF from scratch

### Technologies Used

- **Libraries**: scikit-learn, pandas, numpy, NLTK
- **Approach**: Statistical/rule-based methods
- **Data Structures**: Sparse matrices (CSR format)

### References

#### Tutorials & Guides
- [NLP Pipeline, Ali Alameer | GitHub](https://github.com/Ali-Alameer/NLP/blob/main/week2_pipeline_part1.ipynb)
- [NLP_Getting_started(Preprocessing), Ali H. El-Kassas | Kaggle](https://www.kaggle.com/code/ali01lulu/03-nlp-getting-started-preprocessing/notebook)

#### Libraries & Tools
- [NLTK Documentation](https://www.nltk.org/)
- [PyArabic Documentation](https://github.com/linuxscout/pyarabic) - Arabic text manipulation
- [FarasaPy Documentation](https://github.com/MagedSaeed/farasapy) - Arabic NLP toolkit
- [tnkeeh (تنقيح)](https://github.com/ARBML/tnkeeh) - Arabic preprocessing library
- [qalsadi](https://github.com/linuxscout/qalsadi) - Arabic lemmatizer
- [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) - Advanced Arabic NLP tools
- [ekphrasis](https://github.com/cbaziotis/ekphrasis) - Text processing for social media
- [FlashText](https://pypi.org/project/flashtext/) - Fast keyword replacement/extraction

#### Datasets & Corpora
- [Brown Corpus Overview](https://en.wikipedia.org/wiki/Brown_Corpus)
- [Reuters-21578 Dataset](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html)
- [Common Crawl](https://commoncrawl.org/)
- [Arabic Wikipedia Dumps](https://dumps.wikimedia.org/arwiki/)
- [Arabic 100k Reviews Dataset](https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews)
- [20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)

#### Scikit-learn Documentation
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [LatentDirichletAllocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)

#### Additional Resources
- [Topic modeling visualization guide](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
- [Learn Regex The Easy Way](https://github.com/ziishaned/learn-regex)
- [regexr](https://regexr.com/) - Regular expression editor
- [regexHQ](https://github.com/regexhq) - Community regex patterns


## Module 2: NLP with PyTorch (Deep Learning)

**Module Overview**: This module bridges the gap between statistical NLP and modern deep learning approaches, focusing on tokenization evolution, embeddings, and fine-tuning pre-trained models.

### Key Skills

- Understand tokenization by comparing a manual, from-scratch approach with modern pre-trained tools
- Understand embeddings by visualizing pre-trained models and building one from scratch
- Distinguish how `input_ids` and `attention_mask` relate to tokens and embeddings
- Prepare custom `Dataset` + `DataCollatorWithPadding` for efficient batching
- Fine-tune pre-trained contextual embedding models (like BERT) on custom datasets for higher classification accuracy

### Topics Covered

1. **Transition from Statistical to Neural NLP**:
   - Tokenization evolution: From word-level to subword-level tokenization
   - Vectorization evolution: From sparse (BoW/TF-IDF) to dense (embeddings) representations
   - Static vs. Contextual embeddings: Understanding how modern models handle word meaning
   - The OOV (Out-of-Vocabulary) problem and subword solutions (BPE, WordPiece)

2. **Tokenization**:
   - Manual tokenization from scratch
   - Pre-trained BERT tokenizer (Hugging Face)
   - Subword tokenization and handling OOV words
   - Using `AutoTokenizer` for model compatibility

3. **Embeddings**:
   - Pre-trained GloVe embeddings
   - Visualizing embeddings in 2D space
   - Building embedding models from scratch
   - Static vs. contextual embeddings
   - Contextual embeddings with BERT

4. **Fine-tuning Pre-trained Models**:
   - Loading pre-trained models (DistilBERT)
   - Custom Dataset classes
   - DataCollatorWithPadding for efficient batching
   - Fine-tuning strategies (full model vs. partial)
   - Performance evaluation and generalization

### Lab Structure

- `01_M2_intro.ipynb`: Transition from Statistical to Neural NLP
- `C2_M3_Lab_1_basic_tokenization/`: Tokenization lab
- `C2_M3_Lab_2_embeddings/`: Embeddings lab
- `C2_M3_Lab_4_finetuned_text_classifier/`: Fine-tuning lab

### Stretch Exercises

- `RecSystems.ipynb`: Recommendation systems

### Technologies Used

- **Libraries**: PyTorch, transformers (Hugging Face), sentence-transformers, scikit-learn
- **Models**: BERT, DistilBERT, GloVe
- **Approach**: Deep learning, transfer learning, fine-tuning

### References

#### Courses & Tutorials
- [PyTorch: Techniques and Ecosystem Tools](https://www.coursera.org/learn/pytorch-techniques-and-ecosystem-tools)
- [NLP Demystified](https://www.nlpdemystified.org/course)

#### Models & Libraries
- [BERT Documentation](https://huggingface.co/docs/transformers/en/model_doc/bert)
- [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [DataCollatorWithPadding](https://huggingface.co/docs/transformers/en/main_classes/data_collator#transformers.DataCollatorWithPadding)

#### Datasets
- [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) (Note: this is actually the original, but the lab assumes some pre-processed subset of it. Currently shared via [this link instead](https://drive.google.com/file/d/1NZjBHPzTrLahTaWcZ6GUH1GEYCTAODOw/view?usp=drive_link))


## Module 3: Advanced NLP Applications

**Module Overview**: This module covers practical applications of modern NLP, including task-specific models, zero-shot classification, and large language models.

### Key Skills

- Use pre-trained task-specific models for text classification
- Apply embedding models with classifier heads for custom tasks
- Perform zero-shot classification using Natural Language Inference (NLI)
- Generate text using large language models (LLMs)
- Understand when to use different model types for different tasks

### Topics Covered

1. **Text Classification**:
   - Using task-specific pre-trained models (e.g., sentiment analysis)
   - Using general-purpose embedding models with classifier heads
   - Model selection and performance evaluation
   - Domain-specific model considerations

2. **Natural Language Inference (NLI)**:
   - Understanding NLI: entailment, contradiction, neutral
   - Zero-shot classification using NLI models
   - Multi-label classification
   - Few-shot and zero-shot capabilities of large models

3. **Large Language Models (LLMs)**:
   - Introduction to open LLMs (Phi-3, Mistral, Llama, etc.)
   - Text generation with Phi-3-mini
   - Tokenization and decoding
   - Model loading and inference

### Lab Structure

- `01_Text_Classification.ipynb`: Text classification with pre-trained models
- `02_NLI.ipynb`: Natural Language Inference for zero-shot classification
- `03_LLM.ipynb`: Large Language Models and text generation

### Technologies Used

- **Libraries**: transformers (Hugging Face), sentence-transformers, datasets
- **Models**: RoBERTa, BART, Phi-3-mini, sentence-transformers models
- **Approach**: Pre-trained models, zero-shot learning, text generation

### Hardware Requirements

- Minimum: T4 GPU with 16 GB VRAM (available on free Google Colab)
- For Phi-3-mini: 8 GB VRAM (6 GB with quantization)

### References

#### Models & Libraries
- [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Twitter-RoBERTa-base for Sentiment Analysis](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [BART-large-MNLI](https://huggingface.co/facebook/bart-large-mnli)
- [legal-bert](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
- [biobert](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)

#### Datasets
- [rotten_tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)

#### Resources & Documentation
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding model benchmarks
- [HuggingFace Zero-shot Classification Models](https://huggingface.co/models?pipeline_tag=zero-shot-classification)
- [Natural Language Inference (NLI) Progress](https://nlpprogress.com/english/natural_language_inference.html)
- [SNLI Dataset](https://paperswithcode.com/dataset/snli)
- [MultiNLI Dataset](https://paperswithcode.com/dataset/multinli)
- [SciTail Dataset](https://paperswithcode.com/dataset/scitail)
- [d2l.ai: Natural Language Inference and Dataset](https://d2l.ai/chapter_natural-language-processing-applications/natural-language-inference-and-dataset.html)
- [Zero-shot Classification Task](https://huggingface.co/tasks/zero-shot-classification)
