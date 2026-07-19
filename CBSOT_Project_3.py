
# 
# 
# **Install Libraries**


# !pip install datasets
from datasets import load_dataset

# **Load Dataset**
dataset = load_dataset(
    "CShorten/ML-ArXiv-Papers",
    split="train"
)
print(dataset)
dataset[0]

# **dataset into a Pandas DataFrame.**
import pandas as pd

df = pd.DataFrame(dataset)

df.head()
df.shape
df.columns
df = df[['title', 'abstract']]
df.head()
df = df.head(15000)
df.shape
df.isnull().sum()
df["paper_text"] = df["title"] + " " + df["abstract"]
df[["paper_text"]].head()
type(df[["paper_text"]])
df["paper_text"].head()
type(df["paper_text"])
print(df["paper_text"].iloc[0])
df["paper_text"] = df["paper_text"].str.replace("\n", " ", regex=False)
df["paper_text"] = df["paper_text"].str.strip()
print(df["paper_text"].iloc[0])

# **Sentence Transformers**
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(type(model))
sample_text = df["paper_text"].iloc[0]

embedding = model.encode(sample_text)

print(type(embedding))
print(embedding.shape)
embedding[:10]
sample_embeddings = model.encode(
    df["paper_text"].head(5).tolist()
)

print(type(sample_embeddings))
print(sample_embeddings.shape)
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(
    sample_embeddings[0].reshape(1, -1),
    sample_embeddings[1].reshape(1, -1)
)

print(similarity)
similarity = cosine_similarity(
    sample_embeddings[0].reshape(1,-1),
    sample_embeddings[0].reshape(1,-1)
)

print(similarity)
for i in range(1,5):

    sim = cosine_similarity(
        sample_embeddings[0].reshape(1,-1),
        sample_embeddings[i].reshape(1,-1)
    )

    print(f"Paper 1 vs Paper {i+1}: {sim[0][0]:.4f}")

# **Generate Full Embeddings**
import os
import numpy as np

if os.path.exists("paper_embeddings.npy"):

    print("Loading saved embeddings...")

    embeddings = np.load("paper_embeddings.npy")

else:

    print("Generating embeddings...")

    embeddings = model.encode(
        df["paper_text"].tolist(),
        batch_size=32,
        show_progress_bar=True
    )

    np.save("paper_embeddings.npy", embeddings)

    print("Embeddings saved successfully!")

print(type(embeddings))
print(embeddings.shape)

embeddings.dtype

# !pip install faiss-cpu
import os
import faiss

if os.path.exists("paper_faiss.index"):

    print("Loading existing FAISS index...")

    index = faiss.read_index("paper_faiss.index")

else:

    print("Creating new FAISS index...")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(384)

    index.add(embeddings)

    faiss.write_index(index, "paper_faiss.index")

    print("FAISS index saved successfully!")
import numpy as np

np.linalg.norm(embeddings[0])
print(index.ntotal)
query = "deep learning for medical image analysis"
query_embedding = model.encode([query])
query_embedding.shape

faiss.normalize_L2(query_embedding)


np.linalg.norm(query_embedding[0])
D, I = index.search(query_embedding, 5)

print(D)
print(I)
print(df.iloc[10466]["title"])
print(df.iloc[10466]["abstract"])
def search_papers(query, k=5):

    query_embedding = model.encode([query])

    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k)

    return D, I
D, I = search_papers(
    "deep learning for medical image analysis"
)

print(D)
print(I)
def search_papers(query, k=5):

    query_embedding = model.encode([query])

    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k)

    for score, idx in zip(D[0], I[0]):

        print("=" * 100)

        print("Similarity:", round(float(score), 4))

        print("Title:", df.iloc[idx]["title"])

        print()

        print("Abstract:")

        print(df.iloc[idx]["abstract"][:500])

        print()
search_papers(
    "deep learning for medical image analysis"
)
# !pip install \
transformers==4.46.3 \
huggingface_hub==0.26.2 \
tokenizers==0.20.3 \
sentence-transformers==3.3.1
import transformers
import huggingface_hub

print("Transformers:", transformers.__version__)
print("Transformers Path:", transformers.__file__)

print("Hub:", huggingface_hub.__version__)
print("Hub Path:", huggingface_hub.__file__)

import transformers.utils as u

print(hasattr(u, "HUGGINGFACE_CO_RESOLVE_ENDPOINT"))
import transformers.utils as u

print(hasattr(u, "HUGGINGFACE_CO_RESOLVE_ENDPOINT"))
from transformers import pipeline
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)
summary = summarizer(
    df.iloc[10466]["abstract"],
    max_length=120,
    min_length=40,
    do_sample=False
)
print(summary)
print(summary[0]["summary_text"])
for score, idx in zip(D[0], I[0]):

    print("=" * 100)

    print("Similarity:", round(float(score), 4))

    print("Title:", df.iloc[idx]["title"])

    print()

    summary = summarizer(
        df.iloc[idx]["abstract"],
        max_length=120,
        min_length=40,
        do_sample=False
    )

    print("Summary:")

    print(summary[0]["summary_text"])

    print()
def search_and_summarize(query, k=5):

    query_embedding = model.encode([query])

    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k)

    for score, idx in zip(D[0], I[0]):

        print("=" * 100)

        print("Similarity:", round(float(score), 4))

        print("Title:", df.iloc[idx]["title"])

        print()

        summary = summarizer(
            df.iloc[idx]["abstract"],
            max_length=120,
            min_length=40,
            do_sample=False
        )

        print("Summary:")

        print(summary[0]["summary_text"])

        print()
# !pip install keybert==0.8.5
from keybert import KeyBERT
kw_model = KeyBERT(model)
text = df.iloc[10466]["abstract"]

keywords = kw_model.extract_keywords(text)
keywords = kw_model.extract_keywords(
    text,
    keyphrase_ngram_range=(1,3),
    stop_words="english",
    top_n=5
)

print(keywords)
def search_and_summarize(query, k=5):

    query_embedding = model.encode([query])

    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k)

    for score, idx in zip(D[0], I[0]):

        print("=" * 100)

        print("Similarity:", round(float(score), 4))

        print("Title:", df.iloc[idx]["title"])

        print()

        summary = summarizer(
            df.iloc[idx]["abstract"],
            max_length=120,
            min_length=40,
            do_sample=False
        )

        print("Summary:")

        print(summary[0]["summary_text"])

        print()

        keywords = kw_model.extract_keywords(
            df.iloc[idx]["abstract"],
            keyphrase_ngram_range=(1,3),
            stop_words="english",
            top_n=5
        )

        print("Keywords:")

        for keyword, score in keywords:
            print("-", keyword)

        print()

# **Hybrid NER Architecture**
ner = pipeline(
    "ner",
    aggregation_strategy="simple"
)
text = """
ResNet50 was trained on ImageNet using PyTorch
at Stanford University.
"""
entities = ner(text)

print(entities)
# !pip install langchain langchain-community langchain-core
# !pip install -U langchain langchain-community langchain-huggingface
from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline(
    pipeline=summarizer
)
# !pip install -U langchain-groq
from langchain_groq import ChatGroq
import os
# Try loading from .env file or environment variables
api_key = os.getenv("api")
if not api_key and os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.strip().startswith("api="):
                val = line.split("=", 1)[1].strip()
                if "your_groq_api_key" not in val and val:
                    api_key = val
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "Groq API key not found. Please set 'api=your_key' in your .env file "
        "or set the GROQ_API_KEY environment variable."
    )
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0
)
response = llm.invoke("Hello, who are you?")
response
response.content
from langchain_core.tools import tool


@tool
def search_and_summarize(query: str, k: int = 5) -> str:
    """
    Search research papers from the FAISS database,
    retrieve the top-k most similar papers,
    summarize each paper using BART,
    and return the results.
    """


    query_embedding = model.encode([query])


    faiss.normalize_L2(query_embedding)


    D, I = index.search(query_embedding, k)


    result = ""

    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):

        paper = df.iloc[idx]


        summary = summarizer(
            paper["abstract"],
            max_length=120,
            min_length=40,
            do_sample=False
        )[0]["summary_text"]


        result += f"Rank: {rank}\n"
        result += f"Similarity Score: {round(float(score),4)}\n"
        result += f"Title: {paper['title']}\n\n"


        result += paper["abstract"] + "\n\n"


        result += summary + "\n\n"

    return result
tools = [search_and_summarize]
from langchain.agents import create_agent
agent = create_agent(
    model=llm,
    tools=tools
)


@tool
def extract_keywords(text: str, top_n: int = 5) -> str:
    """
    Extract the most important keywords from the given text
    using the KeyBERT model.
    """


    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )

    result = "Top Keywords:\n\n"

    for rank, (keyword, score) in enumerate(keywords, start=1):

        result += (
            f"{rank}. {keyword} "
            f"(Relevance Score: {round(score, 4)})\n"
        )

    return result
response = extract_keywords.invoke(
    {
        "text": "Deep Learning for Medical Image Reconstruction",
        "top_n": 5
    }
)

print(response)
tools = [search_and_summarize, extract_keywords]

user_query = "Extract the top 5 keywords from Deep Learning for Medical Image Reconstruction."

llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke(user_query)

print(response)

print(response.tool_calls)

tool_call = response.tool_calls[0]

print(tool_call)
tool_name = tool_call["name"]

print(tool_name)
tool_args = tool_call["args"]

print(tool_args)
if tool_name == "extract_keywords":

    tool_result = extract_keywords.invoke(tool_args)

elif tool_name == "search_and_summarize":

    tool_result = search_and_summarize.invoke(tool_args)

print(tool_result)
from langchain_core.messages import ToolMessage
tool_message = ToolMessage(
    content=tool_result,
    tool_call_id=tool_call["id"]
)
print(tool_message)
from langchain_core.messages import SystemMessage, HumanMessage

final_response = llm.invoke(
    [
        SystemMessage(
            content="""
You are a helpful AI assistant.

Rules:
1. Always use the tool output.
2. Never ignore tool results.
3. Present the complete tool output.
4. Add a short explanation after the tool output if necessary.
"""
        ),

        HumanMessage(content=user_query),

        response,

        tool_message
    ]
)

print(final_response.content)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

final_response = llm.invoke(
[
SystemMessage(
content="""
You are a research assistant.

IMPORTANT RULES

1. Never ignore tool output.

2. Always display every keyword returned by the tool.

3. Do not summarize the keywords.

4. Print the keywords exactly as received.

5. After printing them, give a short explanation.
"""
),

HumanMessage(content=user_query),

response,

tool_message

]
)

print(final_response.content)
agent = create_agent(
    model=llm,
    tools=tools
)

# NEW

# !pip install -U langgraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

import json
from langchain_core.tools import tool
from langchain_core.tools import tool

@tool
def compare_papers(
    paper1: str,
    paper2: str
) -> str:
    """
    Compare two research papers based on their abstracts.
    The tool retrieves the closest matching papers from the FAISS database
    and uses the LLM to compare them.
    """


    embedding1 = model.encode([paper1])

    faiss.normalize_L2(embedding1)

    D1, I1 = index.search(embedding1, 1)

    first_paper = df.iloc[I1[0][0]]


    embedding2 = model.encode([paper2])

    faiss.normalize_L2(embedding2)

    D2, I2 = index.search(embedding2, 1)

    second_paper = df.iloc[I2[0][0]]


    comparison_prompt = f"""
Compare the following two research papers.

Paper 1

Title:
{first_paper['title']}

Abstract:
{first_paper['abstract']}


Paper 2

Title:
{second_paper['title']}

Abstract:
{second_paper['abstract']}


Compare them based on:

1. Research Objective

2. Methodology

3. Key Contributions

4. Advantages

5. Limitations

6. Applications

Present the comparison in a clear table.
"""


    response = llm.invoke(comparison_prompt)

    return response.content

@tool
def search_and_summarize(query: str, k: int = 5) -> str:
    """
    Search research papers from the FAISS database,
    retrieve the top-k most similar papers,
    summarize each paper using BART,
    and return the results as a formatted string.
    """
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k)

    result = f"## Research Results for: '{query}'\n\n"

    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        paper = df.iloc[idx]

        summary = summarizer(
            paper["abstract"],
            max_length=120,
            min_length=40,
            do_sample=False
        )[0]["summary_text"]

        result += f"### Rank {rank}: {paper['title']}\n"
        result += f"- **Similarity Score:** {round(float(score), 4)}\n"
        result += f"- **Abstract:** {paper['abstract'].strip()}\n"
        result += f"- **Summary:** {summary}\n\n"
        result += "---\n\n"

    return result

@tool
def extract_keywords(text: str, top_n: int = 5) -> str:
    """
    Extract the most important keywords from a given text or abstract.
    """
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )

    result = "### Top Extracted Keywords:\n"
    for rank, (keyword, score) in enumerate(keywords, start=1):
        result += f"{rank}. {keyword} (Score: {round(score, 4)})\n"

    return result + "\n"

@tool
def save_report_to_file(content: str, filename: str, mode: str = 'w') -> str:
    """
    Saves a synthesized research report or any text content to a local file.
    Set mode to 'w' to create/overwrite a file, or 'a' to append to an existing file.
    """
    with open(filename, mode, encoding='utf-8') as f:
        f.write(content + "\n\n")
    return f"Successfully saved the report to {filename} using mode '{mode}'."

tools = [search_and_summarize, extract_keywords, save_report_to_file, compare_papers]
llm_with_tools = llm.bind_tools(tools)
def chatbot_node(state: MessagesState):
    system_prompt = (
        "You are an expert research agent. When using tools, summarize the results beautifully "
        "for the user. Always display the data you are writing to a file in the chat response as well."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}
def route_tools(state: MessagesState):
    """If the LLM decides to use a tool, route to the tools node. Otherwise, end."""
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
builder.add_edge("tools", "chatbot")
memory = MemorySaver()
agent_executor = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "research_session_1"}}
response_1 = agent_executor.invoke(
    {"messages": [("user", "Find the top 2 papers on Vision Transformers. Summarize them and save the result to a file named 'vit_report.txt'.")]},
    config=config)

print(response_1["messages"][-1].content)
print("-" * 50)

response_2 = agent_executor.invoke(
    {"messages": [("user", "Actually, extract the keywords for those papers and append them to that same file.")]},
    config=config)

print(response_2["messages"][-1].content)

