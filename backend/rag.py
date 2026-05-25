import os
import logging
from typing import Dict, Any, List
from backend.storage import DatabaseManager
from backend.embeddings import EmbeddingService
from backend.llm import LLMService, LLMServiceException
from backend.retrieval import search_similarity

logger = logging.getLogger(__name__)

def build_rag_prompt(retrieved_context: str, history: str, user_question: str) -> str:
    """
    Builds the complete structured prompt instructions for the LLM.
    Strictly instructs the LLM to only answer based on context.
    """
    return f"""You are GenAI Chat Assistant with RAG's internal support assistant. Your primary task is to answer technical questions and policy inquiries based strictly and exclusively on the retrieved context below.

Rules:
1. Grounding: If the retrieved context does not contain enough information to answer the question, or if there is no context retrieved, you MUST reply with exactly: "I could not find enough information in the knowledge base to answer this question." Do not make up facts or extrapolate.
2. Conversation History: Use the conversation history below to resolve pronouns (e.g. "it", "they") and maintain thread continuity, but do not use external facts.
3. Formatting: Present technical instructions clearly using bold headers or bullet lists where appropriate.

Retrieved Context:
{retrieved_context}

Conversation History:
{history}

Question:
{user_question}

Answer:"""

class RAGService:
    def __init__(self, db: DatabaseManager, embedder: EmbeddingService, llm: LLMService):
        self.db = db
        self.embedder = embedder
        self.llm = llm
        
        # Load configuration overrides from env
        self.top_k = int(os.getenv("TOP_K_CHUNKS", "3"))
        self.threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))

    def answer_question(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Coordinates the Retrieval-Augmented Generation (RAG) workflow.
        """
        logger.info(f"Processing query for session {session_id}: '{message}'")

        # 1. Fetch recent conversation history (last 5 message pairs)
        history_msgs = self.db.get_session_messages(session_id, limit=10)
        history_str = ""
        for msg in history_msgs:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role_label}: {msg['content']}\n"

        # 2. Generate embedding for the user's query
        query_embedding = self.embedder.get_embedding(message, is_query=True)

        # 3. Perform similarity search on SQLite using the retrieval module
        matched_chunks = search_similarity(
            self.db,
            query_embedding, 
            top_k=self.top_k, 
            threshold=self.threshold
        )

        logger.info(f"Retrieved {len(matched_chunks)} chunks above similarity threshold ({self.threshold})")

        # Log similarity scores
        scores = [chunk["similarity"] for chunk in matched_chunks]
        if scores:
            logger.info(f"Similarity scores: {scores}")

        # Grounding: Check if matches were found.
        # If no chunks match the similarity threshold, we immediately return the safe fallback response.
        fallback_reply = "I could not find enough information in the knowledge base to answer this question."
        
        if not matched_chunks:
            # Persist message history even for fallbacks
            self.db.add_message(session_id, "user", message)
            self.db.add_message(
                session_id=session_id,
                role="assistant",
                content=fallback_reply,
                tokens_used=0,
                retrieved_chunks_count=0,
                similarity_scores=[]
            )
            return {
                "reply": fallback_reply,
                "tokensUsed": 0,
                "retrievedChunks": 0,
                "sourceDocuments": []
            }

        # 4. Construct context string from matched chunks
        context_parts = []
        for idx, chunk in enumerate(matched_chunks):
            context_parts.append(
                f"Source Document: {chunk['document_title']}\n"
                f"Content: {chunk['text']}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        # 5. Build prompt using template
        prompt = build_rag_prompt(
            retrieved_context=context_str,
            history=history_str,
            user_question=message
        )

        # 6. Pass prompt to LLM
        try:
            reply, tokens_used = self.llm.generate_response(prompt)
            
            # Grounding check: double check if LLM returned the fallback reply
            reply_stripped = reply.strip().strip('"').strip("'").rstrip(".")
            fallback_stripped = fallback_reply.rstrip(".")
            
            is_fallback = reply_stripped == fallback_stripped
            
            # 7. Persist interaction to database
            self.db.add_message(session_id, "user", message)
            self.db.add_message(
                session_id=session_id,
                role="assistant",
                content=reply,
                tokens_used=tokens_used,
                retrieved_chunks_count=0 if is_fallback else len(matched_chunks),
                similarity_scores=[] if is_fallback else scores
            )

            return {
                "reply": reply,
                "tokensUsed": tokens_used,
                "retrievedChunks": 0 if is_fallback else len(matched_chunks),
                "sourceDocuments": [] if is_fallback else matched_chunks
            }

        except LLMServiceException as e:
            # Re-raise standard API exceptions to be formatted by FastAPI
            raise e
        except Exception as e:
            logger.error(f"Error executing RAG pipeline: {e}")
            raise LLMServiceException(f"Failed to generate response: {str(e)}", status_code=500)
