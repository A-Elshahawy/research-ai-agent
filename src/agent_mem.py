import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

load_dotenv()


class AgentMemory:
    """Memory system for the research agent."""

    def __init__(self, max_tokens=4000):
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.research_memory = []
        self.max_tokens = max_tokens
        self.embeddings = OpenAIEmbeddings()

    def add_research_item(self, item: Dict[str, Any]):
        """Add a research item to memory."""
        # Add timestamp
        item["timestamp"] = time.time()
        self.research_memory.append(item)

    def get_relevant_research(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get research items most relevant to the query."""
        if not self.research_memory:
            return []

        # Create embeddings
        query_embedding = self.embeddings.embed_query(query)

        # Get embeddings for research items if they don't have one
        for item in self.research_memory:
            if "embedding" not in item:
                item["embedding"] = self.embeddings.embed_query(
                    f"{item.get('title', '')}: {item.get('content', '')}"
                )

        # Calculate similarity
        similarities = []
        for item in self.research_memory:
            similarity = self._cosine_similarity(query_embedding, item["embedding"])
            similarities.append((similarity, item))

        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in similarities[:k]]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm_a * norm_b)

    def get_conversation_history(self) -> str:
        """Get formatted conversation history."""
        return self.conversation_memory.buffer

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        if role.lower() == "human":
            self.conversation_memory.chat_memory.add_user_message(content)
        elif role.lower() == "ai":
            self.conversation_memory.chat_memory.add_ai_message(content)
        elif role.lower() == "system":
            self.conversation_memory.chat_memory.add_message(
                SystemMessage(content=content)
            )

    def get_memory_context(self, query: str = "") -> str:
        """Get context from memory for the agent."""
        relevant_research = self.get_relevant_research(query)

        memory_context = "Previous research findings:\n"
        for i, item in enumerate(relevant_research):
            memory_context += (
                f"{i + 1}. {item.get('title', 'Untitled')}: {item.get('content', '')}\n"
            )

        return memory_context
