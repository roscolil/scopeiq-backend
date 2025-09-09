from typing import Dict, Any, List
from src.ai.graph import graph


class ChatService:
    def __init__(self):
        pass

    async def process_chat_query(
        self,
        query: str,
        project_id: str = None,
        document_id: str = None,
        conversation_history: List = None,
        context_type: str = "document",
        include_search_results: bool = True,
    ) -> Dict[str, Any]:
        """Process chat query with document context"""

        response = await graph.ainvoke({"question": query, "project_id": project_id})
        if include_search_results:
            retrieved_docs = response["context"]
        else:
            retrieved_docs = []
        answer = response["answer"]

        # Extract source information
        sources_used = []
        for doc in retrieved_docs:
            if "chunk_id" in doc.metadata:
                sources_used.append(doc.metadata["chunk_id"])

        return {"response": answer, "metadata": {"sources_used": sources_used}}


# Global service instance
chat_service = ChatService()
