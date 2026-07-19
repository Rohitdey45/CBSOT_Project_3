# 🤖 Autonomous Agentic AI Research System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Agent_Framework-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-red.svg)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-black.svg)](https://groq.com/)

> **Final Capstone Project of the CBSOT Internship**

An **Agentic AI Research Assistant** that autonomously discovers, analyzes, summarizes, and compares research papers through intelligent tool selection and conversational reasoning.

Unlike traditional RAG systems that simply retrieve documents, this project leverages **LangGraph**, **LangChain**, and **Groq Llama 3.3 70B** to build a **stateful AI agent** capable of deciding which tools to use, maintaining conversation memory, performing semantic search, extracting keywords, comparing papers, and generating structured outputs.

---

# ✨ Features

* 🧠 Stateful conversation memory using MemorySaver
* 🔍 Semantic research paper retrieval using FAISS
* 📚 Dense embeddings with Sentence Transformers
* 📝 AI-powered paper summarization
* 🏷️ Automatic keyword extraction using KeyBERT
* 📊 Intelligent comparison between research papers
* 🛠️ Custom LangChain tools with tool calling
* 💬 Multi-turn conversational interface
* ⚡ High-speed inference using Groq Llama 3.3 70B
* 📄 Structured report generation

---

# System Architecture

```text
                     User Query
                          │
                          ▼
              LangGraph Agent Workflow
                          │
             Conversation Memory
                (MemorySaver)
                          │
                          ▼
                Groq Llama 3.3 70B
                          │
                Autonomous Reasoning
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
   Semantic Search   Keyword Tool   Paper Comparison
      (FAISS)          (KeyBERT)        (LLM)
          │               │                │
          └───────────────┼────────────────┘
                          │
                          ▼
               Response Generation
                          │
                          ▼
                 Final AI Response
```

---

# 🚀 Tech Stack

### AI & LLM

* Groq API
* Llama 3.3 70B Versatile
* LangChain
* LangGraph

### NLP

* Sentence Transformers
* Hugging Face Transformers
* KeyBERT

### Retrieval

* FAISS Vector Database
* Dense Embeddings
* Semantic Search

### Development

* Python
* Pandas
* NumPy
* Jupyter Notebook

---

# 📦 Installation

### Clone the repository

```bash
git clone https://github.com/Rohitdey45/CBSOT_Project_3.git

cd CBSOT_Project_3
```

### Install dependencies

```bash
pip install -r Requirements.txt
```

### Create a `.env` file

```env
api = your_groq_api_key
```

### Run

Open the Jupyter Notebook and execute the cells sequentially.

---

# 🎯 Learning Outcomes

This project helped me gain practical experience with:

* Agentic AI Systems
* LangGraph Workflows
* LangChain Tool Calling
* Memory-based AI Agents
* Retrieval-Augmented Generation (RAG)
* Vector Databases (FAISS)
* Semantic Search
* Research Paper Intelligence Systems
* Prompt Engineering
* Large Language Models

---

# 🏆 Internship

This project was developed as the **third and final capstone project** of the **CBSOT Agentic AI Internship**.

The objective was to build a practical Agentic AI application capable of autonomous reasoning, intelligent tool usage, conversational memory, and semantic information retrieval using modern LLM frameworks.

---

# 🙏 Acknowledgements

A heartfelt thanks to **CBSOT** and all the mentors for their continuous guidance and support throughout the internship. This capstone project provided an opportunity to explore modern Agentic AI architectures and apply them to solving real-world research challenges.

---

## 👨‍💻 Author

**Rohit Dey** 

---

## ⭐ If you found this project interesting, consider giving it a star!
