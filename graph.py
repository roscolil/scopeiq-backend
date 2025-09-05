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

from ingest import init_pinecone


def init_retriever(pc_index):

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=pc_index, embedding=embeddings)
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.4},
    )


def init_llm():
    # Check if OpenAI API key is set
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set this environment variable to use the LLM functionality.")
        return None

    return init_chat_model(model="gpt-4o", model_provider="openai")


pc_index = init_pinecone()
retriever = init_retriever(pc_index)
rag_llm = init_llm()

rag_system_prompt = """You are an AI assistant specialized in construction and jobsite document analysis.

# Instructions
- Answer strictly in concise points, no explanations or introductory text. 
- Do not repeat or paraphrase the question.
- Do not add context or commentary.
- For questions that are ambiguous, provide all related information you can find regarding material type, specifications or measurements.
- For questions about the number of items, provide exact numbers based on the structured analysis results.
- For dimensional queries, reference specific measurements and specifications found in the analysis.
- For spatial queries (adjacent, inside, north, south, etc), use the spatial relationship information to provide accurate location-based answers.
- Base your answer on the specific information provided in the context.

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
    answer: str


# Define application steps
def retrieve(state: State):
    print("--retrieve--")
    if retriever is None:
        print("Warning: Retriever is not initialized. Returning empty context.")
        return {"context": []}
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    print("--generate--")
    if rag_llm is None:
        return {
            "answer": "Error: LLM is not initialized. Please set OPENAI_API_KEY environment variable."
        }

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response = rag_llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
