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
# # Day 1b - Evaluation and Structured Output
#
# This tutorial builds on the prompting techniques from `day-1a-prompting.py` and focuses on evaluating LLM outputs and using structured output formats using OpenRouter AI.
#
# ## Learning Objectives
#
# By the end of this tutorial, you will be able to:
# - Evaluate LLM outputs using pointwise and pairwise evaluation methods
# - Use structured output formats for programmatic evaluation
# - Understand best practices and limitations of LLM evaluation
# - Build evaluation systems for real-world applications
#
# ## Prerequisites
#
# Before starting, make sure you have:
# - Completed `day-1a-prompting.py` (recommended but not required)
# - Completed the setup instructions in `SETUP.md`
# - Obtained an OpenRouter API key from [OpenRouter](https://openrouter.ai/keys)
# - Installed the required dependencies (see `SETUP.md`)

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Import the SDK and Helpers

# %%
from openai import OpenAI
from IPython.display import Markdown, display
import os

# %% [markdown]
# ### Set Up Retry Helper
#
# This allows you to run all cells without worrying about per-minute quota limits.
# The retry helper will automatically retry requests that fail due to rate limiting (429) or service unavailability (503).
# Note: OpenAI SDK has built-in retry logic, but you can add custom retry handling if needed.

# %%
# OpenAI SDK has built-in retry logic for rate limits and service errors

# %% [markdown]
# ### Initialize the Client
#
# OpenRouter provides a unified API that gives you access to hundreds of AI models through a single endpoint.
# We use the OpenAI SDK with OpenRouter's base URL for compatibility.

# %%
# Get API key from environment variable or use the commented line for Colab
# For Colab: import google.colab.userdata; api_key = google.colab.userdata.get('OPENROUTER_API_KEY')
api_key = os.getenv('OPENROUTER_API_KEY')
if not api_key:
    raise ValueError("Please set OPENROUTER_API_KEY environment variable. Get your key from https://openrouter.ai/keys")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Default model to use
# You can use any model available on OpenRouter.
# Check https://openrouter.ai/models for available models.
DEFAULT_MODEL = "deepseek/deepseek-chat"

# %% [markdown]
# **Note:** You can use any model available on OpenRouter. Check https://openrouter.ai/models for available models.

# %% [markdown]
# ## Part 1: Evaluation and Structured Output
#
# When using LLMs in real-world applications, it's important to understand how well they are performing.
# The open-ended generation capabilities of LLMs can make evaluation challenging.
# This section covers techniques for evaluating LLM outputs.

# %% [markdown]
# ### Document Summarization Example
#
# For this evaluation example, we'll use a document summarization task.
# First, let's download a sample document and upload it to the Gemini API.
#
# Note: OpenRouter doesn't support file uploads directly like Gemini.
# For document processing, you would need to:
# 1. Extract text from the PDF (using libraries like PyPDF2 or pdfplumber)
# 2. Include the text in the prompt, or
# 3. Use a model that supports vision/document understanding with base64 encoding
# For this example, we'll read the PDF text and include it in prompts

# %% [markdown]
# ### Summarize a Document

# %%
def answer_question(document: str, question: str) -> str:
    """Execute the question on the document."""
    # Note: For simplicity, we'll use a placeholder approach.
    # In practice, you'd extract text from the PDF and include it in the prompt.
    # For this example, we'll simulate document context.
    prompt = (
        "Based on the following document context, answer the question."
        f"\nDocument: {document}"
        f"\nQuestion: {question}"
        "\nAnswer:"
    )

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=250,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )
    
    return response.choices[0].message.content


doc = (
    "Renewable Energy Policy Framework 2024-2030\n"
    "Executive Summary\n"
    "This comprehensive policy document outlines the national strategy for transitioning to renewable energy sources over the next decade. "
    "The framework addresses solar, wind, hydroelectric, and geothermal energy initiatives, with specific targets and implementation timelines.\n\n"
    "Chapter 1: Solar Energy Initiatives\n"
    "The solar energy program aims to install 50 gigawatts of photovoltaic capacity by 2030. "
    "Key initiatives include rooftop solar subsidies for residential properties, large-scale solar farms in desert regions, "
    "and integration with smart grid technologies. The program includes tax incentives for commercial installations and "
    "streamlined permitting processes for solar projects exceeding 1 megawatt capacity.\n\n"
    "Chapter 2: Wind Energy Development\n"
    "Wind energy targets include 30 gigawatts of onshore and offshore wind capacity. Priority regions have been identified "
    "based on wind speed data collected over the past five years. Offshore wind farms will be developed in three phases, "
    "with the first phase expected to generate 5 gigawatts by 2026. Onshore projects will focus on rural areas with minimal "
    "environmental impact, requiring environmental impact assessments before approval.\n\n"
    "Chapter 3: Hydroelectric Power Expansion\n"
    "Existing hydroelectric facilities will be upgraded to increase efficiency by 15%. New small-scale hydroelectric projects "
    "will be developed in mountainous regions, with a target of 10 gigawatts additional capacity. These projects must comply "
    "with strict environmental regulations to protect aquatic ecosystems and fish migration patterns.\n\n"
    "Chapter 4: Geothermal Energy Exploration\n"
    "Geothermal energy represents an underutilized resource with significant potential. The policy allocates $2 billion for "
    "geothermal exploration and development over the next six years. Initial focus areas include regions with known geothermal "
    "activity, where drilling operations will assess resource viability. Successful projects will receive government funding "
    "covering up to 40% of development costs.\n\n"
    "Chapter 5: Grid Modernization and Storage\n"
    "Modernizing the electrical grid is essential for integrating renewable sources. The policy mandates smart grid upgrades "
    "in all major metropolitan areas by 2028. Battery storage systems will be deployed at renewable energy sites to address "
    "intermittency issues. The target is 20 gigawatt-hours of storage capacity, utilizing both lithium-ion and emerging "
    "technologies like flow batteries and compressed air energy storage.\n\n"
    "Chapter 6: Economic Incentives and Funding\n"
    "The government will provide $50 billion in funding through various mechanisms: direct grants, low-interest loans, tax credits, "
    "and public-private partnerships. Small businesses and residential customers can access rebates covering 30% of installation "
    "costs. Large-scale projects may qualify for accelerated depreciation schedules and reduced corporate tax rates.\n\n"
    "Chapter 7: Workforce Development\n"
    "Training programs will be established to develop a skilled workforce for the renewable energy sector. Technical colleges "
    "will receive funding to expand renewable energy curricula. Apprenticeship programs will connect 10,000 workers with "
    "renewable energy companies over the next five years. Certification programs will be standardized across all regions.\n\n"
    "Chapter 8: Environmental Impact and Regulations\n"
    "All renewable energy projects must undergo comprehensive environmental impact assessments. Projects affecting protected "
    "wildlife habitats require additional mitigation measures. The policy establishes a carbon offset program where renewable "
    "energy generation can be traded. Strict monitoring and reporting requirements ensure compliance with environmental standards.\n\n"
    "Chapter 9: International Cooperation\n"
    "The framework includes provisions for international partnerships and technology transfer. Bilateral agreements with leading "
    "renewable energy nations will facilitate knowledge sharing and joint research initiatives. The country will participate in "
    "global climate initiatives and contribute to international renewable energy standards development.\n\n"
    "Chapter 10: Implementation Timeline and Milestones\n"
    "Phase 1 (2024-2026): Foundation and pilot projects. Establish regulatory framework, begin large-scale solar installations, "
    "initiate wind farm development, and launch workforce training programs.\n"
    "Phase 2 (2027-2028): Scaling operations. Complete 60% of solar targets, operationalize first offshore wind farms, "
    "deploy smart grid infrastructure, and expand storage capacity.\n"
    "Phase 3 (2029-2030): Full deployment and optimization. Achieve all capacity targets, complete grid modernization, "
    "establish international partnerships, and evaluate program effectiveness for future planning.\n\n"
    "Conclusion\n"
    "This renewable energy policy framework represents a comprehensive approach to transitioning to sustainable energy sources. "
    "Success requires coordination between government agencies, private sector investment, technological innovation, and public "
    "support. Regular reviews and adjustments will ensure the policy remains effective and responsive to changing circumstances."
)

question = 'What are the main renewable energy sources covered in this policy, and what are the capacity targets for each?'
answer = answer_question(document=doc, question=question)
Markdown(answer)

# %% [markdown]
# ### Define an Evaluator
#
# For evaluation tasks, you may wish to assess various aspects:
# - **Instruction following**: How well the model followed the prompt
# - **Groundedness**: Whether the response contains only information from the provided context
# - **Fluency**: How easy the text is to read
# - **Conciseness**: Whether the response is appropriately brief
# - **Quality**: Overall quality of the response
#
# You can instruct an LLM to perform these evaluations similar to how you would instruct a human rater: with a clear definition and assessment rubric.

# %%
import enum

# Define a structured enum class to capture the evaluation result for QA
class QARating(enum.Enum):
    VERY_GOOD = '5'   # Very good answer: fully correct, fully grounded, fully relevant, clear, and complete
    GOOD = '4'        # Good answer: nearly perfect, minor flaws
    OK = '3'          # Ok answer: somewhat correct or relevant, possibly incomplete or unclear
    BAD = '2'         # Bad answer: partially or mostly incorrect or missing, problematic grounding
    VERY_BAD = '1'    # Very bad: entirely wrong, hallucinated, or off-topic

# Adapted QA evaluation prompt
QA_EVAL_PROMPT = """\
# Instruction
You are an expert evaluator. Your task is to evaluate the quality of answers to questions, given a context.
We will provide you with a question, a reference/context (if given), and an AI-generated answer.
You should first consider the question and context, then evaluate the answer based on the Criteria in the Evaluation section below.
Assign a rating following the Rating Rubric and Evaluation Steps. Give step-by-step explanations for your rating, and only choose ratings from the Rating Rubric.

# Evaluation
## Metric Definition
You will be assessing question answering quality, measuring the model's ability to provide accurate, complete, grounded, and well-communicated answers. The answer should be based only on the provided context (if any) and should directly respond to the question.

## Criteria
Instruction following: The answer directly and fully addresses the question asked, following any stated instructions (e.g. length requirement, citation, detail).
Groundedness: The answer is based solely on the provided context or reference (if any); it does not make claims not in the context.
Correctness: The answer is factually accurate as far as can be determined from the context.
Completeness & Relevance: The answer covers all key components of the question without excessive or irrelevant information.
Clarity: The answer is easy to read and understand.

## Rating Rubric
5: (Very good). The answer follows instructions, is grounded, correct, complete, and clear.
4: (Good). The answer is nearly perfect minor flaw(s), fully or mostly satisfies all requirements.
3: (Ok). The answer is partially correct/good, but is incomplete or unclear.
2: (Bad). The answer is substantially incomplete or incorrect, or has major grounding issues.
1: (Very bad). The answer misses the point, is ungrounded, or entirely incorrect.

## Evaluation Steps
STEP 1: Assess the answer by the criteria: instruction following, groundedness, correctness, completeness, and clarity.
STEP 2: Assign a rating based on the rubric.

# Inputs and AI-generated Answer
## Question
{question}

## Context (if any)
{context}

## AI-generated Answer
{answer}
"""

def eval_qa(question, answer, context=""):
    """Evaluate a QA answer given the input question and an optional context."""
    messages = []
    messages.append({
        "role": "user",
        "content": QA_EVAL_PROMPT.format(
            question=question,
            context=context if context else "<No context provided. Answer must be generic or rely only on the question.>",
            answer=answer
        )
    })
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
    )
    verbose_eval = response.choices[0].message.content
    messages.append({
        "role": "assistant",
        "content": verbose_eval
    })

    # Structure conversion request
    messages.append({
        "role": "user",
        "content": "Convert the final score to one of: 5, 4, 3, 2, or 1. Respond with only the number."
    })
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=5,
    )
    score_text = response.choices[0].message.content.strip()
    try:
        structured_eval = QARating(score_text)
    except ValueError:
        import re
        match = re.search(r'\b([1-5])\b', score_text)
        if match:
            structured_eval = QARating(match.group(1))
        else:
            structured_eval = QARating.OK  # Default fallback

    return verbose_eval, structured_eval

# Usage example for QA:
text_eval, struct_eval = eval_qa(
    question=question,   # The question about renewable energy
    answer=answer,       # The model's generated answer
    context=doc          # The full document context
)
Markdown(text_eval)

# %%
struct_eval

# %% [markdown]
# In this example, the model generated a textual justification in a chat context. This full text response is useful for human interpretation and gives the model a place to "collect notes" while assessing the text. The working "notes" are used when generating the final result output.
#
# In the next turn, the model converts the text output into a structured response. If you want to aggregate scores or use them programmatically, you want to avoid parsing unstructured text. Here the `SummaryRating` schema is passed, so the model converts the chat history into an instance of the `SummaryRating` enum.

# %% [markdown]
# ### Pointwise Evaluation
#
# The technique used above, where you evaluate a single input/output pair against some criteria, is known as **pointwise evaluation**. This is useful for evaluating singular outputs in an absolute sense, such as "was it good or bad?"
#
# In this exercise, you'll try different guidance prompts with a set of questions:

# %%
import functools

# Try these instructions, or edit and add your own
terse_guidance = "Answer the following question in a single sentence, or as close to that as possible."
moderate_guidance = "Provide a brief answer to the following question, use a citation if necessary, but only enough to answer the question."
cited_guidance = "Provide a thorough, detailed answer to the following question, citing the document and supplying additional background information as much as possible."

guidance_options = {
    'Terse': terse_guidance,
    'Moderate': moderate_guidance,
    'Cited': cited_guidance,
}

questions = [
    "What is the total funding allocated for renewable energy initiatives?",
    "What are the three implementation phases and their timeframes?",
]

@functools.cache
def answer_question(question: str, guidance: str = '', document_context: str = '') -> str:
    """Generate an answer to the question using the document and guidance."""
    messages = []
    if guidance:
        messages.append({
            "role": "system",
            "content": guidance
        })
    
    # Use the actual document context
    prompt = f"""Based on the following document context, answer the question.
    
Document: {document_context if document_context else doc}
Question: {question}

Answer:"""
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.0
    )
    
    return response.choices[0].message.content

answer = answer_question(questions[0], terse_guidance, document_context=doc)
Markdown(answer)

# %% [markdown]
# Now set up a question-answering evaluator:

# %%
QA_PROMPT = """\
# Instruction
You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models.
We will provide you with the user prompt and an AI-generated response.
You should first read the user prompt carefully for analyzing the task, and then evaluate the quality of the response based on and rules provided in the Evaluation section below.

# Evaluation
## Metric Definition
You will be assessing question answering quality, which measures the overall quality of the answer to the question in the user prompt. Pay special attention to length constraints, such as in X words or in Y sentences. The instruction for performing a question-answering task is provided in the user prompt. The response should not contain information that is not present in the context (if it is provided).

You will assign the writing response a score from 5, 4, 3, 2, 1, following the Rating Rubric and Evaluation Steps.
Give step-by-step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.

## Criteria Definition
Instruction following: The response demonstrates a clear understanding of the question answering task instructions, satisfying all of the instruction's requirements.
Groundedness: The response contains information included only in the context if the context is present in the user prompt. The response does not reference any outside information.
Completeness: The response completely answers the question with sufficient detail.
Fluent: The response is well-organized and easy to read.

## Rating Rubric
5: (Very good). The answer follows instructions, is grounded, complete, and fluent.
4: (Good). The answer follows instructions, is grounded, complete, but is not very fluent.
3: (Ok). The answer mostly follows instructions, is grounded, answers the question partially and is not very fluent.
2: (Bad). The answer does not follow the instructions very well, is incomplete or not fully grounded.
1: (Very bad). The answer does not follow the instructions, is wrong and not grounded.

## Evaluation Steps
STEP 1: Assess the response in aspects of instruction following, groundedness, completeness, and fluency according to the criteria.
STEP 2: Score based on the rubric.

# User Inputs and AI-generated Response
## User Inputs
### Prompt
{prompt}

## AI-generated Response
{response}
"""

class AnswerRating(enum.Enum):
    VERY_GOOD = '5'
    GOOD = '4'
    OK = '3'
    BAD = '2'
    VERY_BAD = '1'

@functools.cache
def eval_answer(prompt, ai_response, n=1):
    """Evaluate the generated answer against the prompt/question used."""
    messages = []
    
    # Generate the full text response
    messages.append({
        "role": "user",
        "content": QA_PROMPT.format(prompt=prompt, response=ai_response)
    })
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages
    )
    verbose_eval = response.choices[0].message.content
    messages.append({
        "role": "assistant",
        "content": verbose_eval
    })
    
    # Coerce into the desired structure
    messages.append({
        "role": "user",
        "content": "Convert the final score to one of: 5, 4, 3, 2, or 1. Respond with only the number."
    })
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=5
    )
    
    # Parse the score
    score_text = response.choices[0].message.content.strip()
    try:
        structured_eval = AnswerRating(score_text)
    except ValueError:
        # Fallback: try to extract number from text
        import re
        match = re.search(r'\b([1-5])\b', score_text)
        if match:
            structured_eval = AnswerRating(match.group(1))
        else:
            structured_eval = AnswerRating.OK  # Default fallback
    
    return verbose_eval, structured_eval

text_eval, struct_eval = eval_answer(prompt=questions[0], ai_response=answer)
display(Markdown(text_eval))
print(struct_eval)

# %% [markdown]
# Now run the evaluation task in a loop. Note that the guidance instruction is hidden from the evaluation agent. If you passed the guidance prompt, the model would score based on whether it followed that guidance, but for this task the goal is to find the best overall result based on the user's question.

# %%
import collections

# Number of times to repeat each task to reduce error and calculate an average
NUM_ITERATIONS = 1

scores = collections.defaultdict(int)
responses = collections.defaultdict(list)

for question in questions:
    display(Markdown(f'## {question}'))
    for guidance, guide_prompt in guidance_options.items():
        
        for n in range(NUM_ITERATIONS):
            # Generate a response
            answer = answer_question(question, guide_prompt, document_context=doc)
            
            # Evaluate the response (note that the guidance prompt is not passed)
            written_eval, struct_eval = eval_answer(question, answer, n)
            print(f'{guidance}: {struct_eval}')
            
            # Save the numeric score
            scores[guidance] += int(struct_eval.value)
            
            # Save the responses, in case you wish to inspect them
            responses[(guidance, question)].append((answer, written_eval))

# %%
# Aggregate the scores
for guidance, score in scores.items():
    avg_score = score / (NUM_ITERATIONS * len(questions))
    nearest = AnswerRating(str(round(avg_score)))
    print(f'{guidance}: {avg_score:.2f} - {nearest.name}')

# %% [markdown]
# ### Pairwise Evaluation
#
# The pointwise evaluation prompt used in the previous step has 5 levels of grading. This may be too coarse for your system, or perhaps you wish to improve on a prompt that is already "very good."
#
# Another approach is to compare two outputs against each other. This is **pairwise evaluation**, and is a key step in ranking and sorting algorithms, which allows you to use it to rank your prompts either instead of, or in addition to the pointwise approach.

# %%
QA_PAIRWISE_PROMPT = """\
# Instruction
You are an expert evaluator. Your task is to evaluate the quality of the responses generated by two AI models. We will provide you with the user input and a pair of AI-generated responses (Response A and Response B). You should first read the user input carefully for analyzing the task, and then evaluate the quality of the responses based on the Criteria provided in the Evaluation section below.

You will first judge responses individually, following the Rating Rubric and Evaluation Steps. Then you will give step-by-step explanations for your judgment, compare results to declare the winner based on the Rating Rubric and Evaluation Steps.

# Evaluation
## Metric Definition
You will be assessing question answering quality, which measures the overall quality of the answer to the question in the user prompt. Pay special attention to length constraints, such as in X words or in Y sentences. The instruction for performing a question-answering task is provided in the user prompt. The response should not contain information that is not present in the context (if it is provided).

## Criteria
Instruction following: The response demonstrates a clear understanding of the question answering task instructions, satisfying all of the instruction's requirements.
Groundedness: The response contains information included only in the context if the context is present in the user prompt. The response does not reference any outside information.
Completeness: The response completely answers the question with sufficient detail.
Fluent: The response is well-organized and easy to read.

## Rating Rubric
"A": Response A answers the given question as per the criteria better than response B.
"SAME": Response A and B answers the given question equally well as per the criteria.
"B": Response B answers the given question as per the criteria better than response A.

## Evaluation Steps
STEP 1: Analyze Response A based on the question answering quality criteria: Determine how well Response A fulfills the user requirements, is grounded in the context, is complete and fluent, and provides assessment according to the criterion.
STEP 2: Analyze Response B based on the question answering quality criteria: Determine how well Response B fulfills the user requirements, is grounded in the context, is complete and fluent, and provides assessment according to the criterion.
STEP 3: Compare the overall performance of Response A and Response B based on your analyses and assessment.
STEP 4: Output your preference of "A", "SAME" or "B" to the pairwise_choice field according to the Rating Rubric.
STEP 5: Output your assessment reasoning in the explanation field.

# User Inputs and AI-generated Responses
## User Inputs
### Prompt
{prompt}

# AI-generated Response

### Response A
{baseline_model_response}

### Response B
{response}
"""

class AnswerComparison(enum.Enum):
    A = 'A'
    SAME = 'SAME'
    B = 'B'

@functools.cache
def eval_pairwise(prompt, response_a, response_b, n=1):
    """Determine the better of two answers to the same prompt."""
    
    messages = []
    
    # Generate the full text response
    messages.append({
        "role": "user",
        "content": QA_PAIRWISE_PROMPT.format(
            prompt=prompt,
            baseline_model_response=response_a,
            response=response_b)
    })
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages
    )
    verbose_eval = response.choices[0].message.content
    messages.append({
        "role": "assistant",
        "content": verbose_eval
    })
    
    # Coerce into the desired structure
    messages.append({
        "role": "user",
        "content": "Convert the final preference to one of: A, SAME, or B. Respond with only the choice."
    })
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=10
    )
    
    # Parse the comparison
    choice_text = response.choices[0].message.content.strip().upper()
    try:
        structured_eval = AnswerComparison(choice_text)
    except ValueError:
        # Fallback: try to match
        if "A" in choice_text and "B" not in choice_text:
            structured_eval = AnswerComparison.A
        elif "B" in choice_text and "A" not in choice_text:
            structured_eval = AnswerComparison.B
        elif "SAME" in choice_text:
            structured_eval = AnswerComparison.SAME
        else:
            structured_eval = AnswerComparison.SAME  # Default fallback
    
    return verbose_eval, structured_eval

question = questions[0]
answer_a = answer_question(question, terse_guidance, document_context=doc)
answer_b = answer_question(question, cited_guidance, document_context=doc)

text_eval, struct_eval = eval_pairwise(
    prompt=question,
    response_a=answer_a,
    response_b=answer_b,
)

display(Markdown(text_eval))
print(struct_eval)

# %% [markdown]
# ## Part 2: Best Practices and Limitations
#
# ### LLM Limitations
#
# LLMs are known to have problems on certain tasks, and these challenges still persist when using LLMs as evaluators. For example:
# - LLMs can struggle with numerical problems (like counting characters in a word)
# - They may not accurately evaluate tasks that require precise measurements
# - They can be biased based on their training data
#
# There are solutions available in some cases, such as connecting tools to handle problems unsuitable to a language model, but it's important that you understand possible limitations and include human evaluators to calibrate your evaluation system and determine a baseline.
#
# One reason that LLM evaluators work well is that all of the information they need is available in the input context, so the model only needs to attend to that information to produce the result. When customizing evaluation prompts, or building your own systems, keep this in mind and ensure that you are not relying on "internal knowledge" from the model, or behavior that might be better provided from a tool.

# %% [markdown]
# ### Improving Confidence
#
# One way to improve the confidence of your evaluations is to include a diverse set of evaluators. That is, use the same prompts and outputs, but execute them on different models, like Gemini Flash and Pro, or even across different providers. This follows the same idea used earlier, where repeating trials to gather multiple "opinions" helps to reduce error, except by using different models the "opinions" will be more diverse.

# %% [markdown]
# ## Summary
#
# In this tutorial, you've learned:
#
# 1. **Evaluation Methods**: How to evaluate LLM outputs using pointwise and pairwise evaluation
# 2. **Structured Output**: How to use structured output formats (enums) for programmatic evaluation
# 3. **Evaluation Prompts**: How to design effective evaluation prompts with clear criteria and rubrics
# 4. **Best Practices**: Understanding limitations and improving evaluation confidence
