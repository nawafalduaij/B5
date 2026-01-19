# Building Generative AI Applications

**Objective**: The student is able to build AI applications to solve problems, beyond manual copy-pasting into commonly available chat interfaces like ChatGPT and Gemini.

## 8-Day AI Engineering Course Outline

### Day 1: Foundations
*   **Prompting** (`day_1a_prompting.ipynb`)
    *   Zero-Shot Prompting
    *   Few-Shot Prompting
    *   Chain-of-Thought Prompting
    *   System Instructions
    *   Code Generation
*   **Evaluation and Structured Output** (`day_1b_evaluation-and-structured-output.ipynb`)
    *   Structured Output
    *   Evaluation Methods
    *   Pointwise Evaluation
    *   Pairwise Evaluation

### Day 2: Embeddings & RAG
*   **Embeddings and Similarity Scores** (`day_2_similarity.ipynb`)
    *   Understanding Embeddings
    *   Semantic Similarity Calculations
*   **Classifying Embeddings with PyTorch** (`day_2_classifier_pytorch.ipynb`)
    *   Classification tasks using embeddings
    *   Building classification models with PyTorch
*   **Document Q&A with RAG** (`day_2_qa_rag.ipynb`)
    *   Retrieval Augmented Generation implementation
    *   Using ChromaDB for vector storage

### Day 3: Agent Architectures
*   **From Prompt to Action** (`day_3a_from_prompt_to_action.ipynb`)
    *   Building your first AI agent with ADK
    *   Agent Development Kit (ADK) setup
    *   Creating agents with tools
*   **Multi-Agent Systems & Workflow Patterns** (`day_3b_agent_architectures.ipynb`)
    *   Sequential Agents
    *   Parallel Agents
    *   Loop Agents
    *   Multi-agent orchestration patterns

### Day 4: Agent Tools & Best Practices
*   **Agent Tools** (`day_4a_agent_tools.ipynb`)
    *   Custom Function Tools
    *   Agent Tools (using agents as tools)
    *   Built-in Code Executor
    *   Tool types overview
*   **Agent Tool Patterns and Best Practices** (`day_4b_agent_tools_best_practices.ipynb`)
    *   Model Context Protocol (MCP) integration
    *   Long-Running Operations (LRO)
    *   Human-in-the-loop approvals
    *   Resumable workflows

### Day 5: Sessions & Memory
*   **Memory Management - Part 1: Sessions** (`day_5a_agent_sessions.ipynb`)
    *   Session Management
    *   Persistent Sessions with DatabaseSessionService
    *   Context Compaction
    *   Session State management
*   **Memory Management - Part 2: Memory** (`day_5b_agent_memory.ipynb`)
    *   Long-term memory with MemoryService
    *   Memory ingestion and retrieval
    *   Automatic memory storage with callbacks
    *   Memory consolidation concepts

### Day 6: Observability & Evaluation
*   **Agent Observability** (`day_6a_agent_observability.ipynb`)
    *   Logs, Traces & Metrics
    *   Debugging with ADK Web UI
    *   LoggingPlugin for production
    *   Custom plugins and callbacks
*   **Agent Evaluation** (`day_6b_agent_evaluation.ipynb`)
    *   Interactive evaluation with ADK Web UI
    *   Systematic evaluation with test cases
    *   Tool trajectory and response metrics
    *   Regression testing

### Day 7: Agent2Agent Communication & Deployment
*   **Agent2Agent (A2A) Communication** (`day_7a_agent2agent_communication.ipynb`)
    *   A2A Protocol overview
    *   Exposing agents via A2A
    *   Consuming remote agents
    *   Cross-framework and cross-organization integration
*   **Agent Deployment** (`day_7b_agent_deployment.ipynb`)
    *   Deploying to Vertex AI Agent Engine
    *   Production-ready agent configuration
    *   Testing deployed agents
    *   Long-term memory with Vertex AI Memory Bank
    *   Cost management and cleanup


## Primary sources

1. https://www.kaggle.com/learn-guide/5-day-genai
2. https://www.kaggle.com/learn-guide/5-day-agents