import os
from pinecone import Pinecone


from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from src.core.config import settings


def init_pinecone():
    """Initialize Pinecone connection"""
    pinecone_api_key = settings.PINECONE_API_KEY
    pinecone_index_name = settings.PINECONE_INDEX_NAME

    if not pinecone_api_key or not pinecone_index_name:
        return None

    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(pinecone_index_name)


def init_retriever(pc_index):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=settings.OPENAI_API_KEY
    )
    vector_store = PineconeVectorStore(index=pc_index, embedding=embeddings)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10},
    )


def init_llm():
    # Check if OpenAI API key is set
    openai_api_key = settings.OPENAI_API_KEY

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set this environment variable to use the LLM functionality.")
        return None

    return init_chat_model(
        model="gpt-4o", model_provider="openai", openai_api_key=openai_api_key
    )


# Initialize components lazily to avoid import-time errors
pc_index = None
retriever = None
rag_llm = None


def get_pc_index():
    global pc_index
    if pc_index is None:
        pc_index = init_pinecone()
    return pc_index


def get_retriever():
    global retriever
    if retriever is None:
        retriever = init_retriever(get_pc_index())
    return retriever


def get_rag_llm():
    global rag_llm
    if rag_llm is None:
        rag_llm = init_llm()
    return rag_llm


rag_system_prompt = """You are an AI assistant specialized in construction and jobsite document analysis.

# Instructions
- Answer strictly in concise points. 
- Do not repeat or paraphrase the question.
- Do not add context or commentary.
- For questions that are ambiguous, provide all related information you can find regarding material type, specifications or measurements.
- For questions about the number of items, provide exact numbers based on the structured analysis results.
- For dimensional queries, reference specific measurements and specifications found in the analysis.
- For spatial queries (adjacent, inside, north, south, etc), use the spatial relationship information to provide accurate location-based answers.
- Base your answer on the specific information provided in the context.
- If the context is empty, please mention it in your answer. Do not make up an answer.

Context: 
{context}"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        ("user", "{question}"),
    ]
)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    project_id: str
    answer: str


# Define application steps
def retrieve(state: State):
    print("--retrieve--")
    current_retriever = get_retriever()
    if current_retriever is None:
        print("Warning: Retriever is not initialized. Returning empty context.")
        return {"context": []}
    retrieved_docs = current_retriever.invoke(
        state["question"], namespace=state["project_id"]
    )
    return {"context": retrieved_docs}


def generate(state: State):
    print("--generate--")
    current_llm = get_rag_llm()
    if current_llm is None:
        return {
            "answer": "Error: LLM is not initialized. Please set OPENAI_API_KEY environment variable."
        }

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response = current_llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
