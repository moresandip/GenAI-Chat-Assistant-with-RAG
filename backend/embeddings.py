import os
import logging
from typing import List
import hashlib
import numpy as np
import google.generativeai as genai
from openai import OpenAI

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        # Configure API clients
        if self.provider == "gemini":
            if self.gemini_key:
                genai.configure(api_key=self.gemini_key)
                logger.info("Gemini Embedding service configured successfully.")
            else:
                logger.warning("Gemini API key is missing. Embedding service will run in offline Mock mode.")
                self.provider = "mock"
        elif self.provider == "openai":
            if self.openai_key:
                self.openai_client = OpenAI(api_key=self.openai_key)
                logger.info("OpenAI Embedding service configured successfully.")
            else:
                logger.warning("OpenAI API key is missing. Embedding service will run in offline Mock mode.")
                self.provider = "mock"
        else:
            logger.info("Embedding service running in Mock (offline) mode.")

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generates vector embedding for the input text using the configured provider.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text string")

        if self.provider == "gemini":
            try:
                # Use models/gemini-embedding-001 for Gemini embeddings
                task_type = "retrieval_query" if is_query else "retrieval_document"
                result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=text,
                    task_type=task_type
                )
                return result["embedding"]
            except Exception as e:
                logger.error(f"Gemini Embedding API failed: {e}. Falling back to mock embeddings.")
                return self._generate_mock_embedding(text)

        elif self.provider == "openai":
            try:
                # Use text-embedding-3-small (1536-dim)
                response = self.openai_client.embeddings.create(
                    input=[text],
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"OpenAI Embedding API failed: {e}. Falling back to mock embeddings.")
                return self._generate_mock_embedding(text)

        else:
            # Fallback to local mock embedding generator
            return self._generate_mock_embedding(text)

    def _generate_mock_embedding(self, text: str, dimension: int = 768) -> List[float]:
        """
        Generates a deterministic unit vector representation of a string.
        Words that are identical map to identical indices in the vector, allowing
        simple similarity calculations to function in offline development environment.
        """
        words = text.lower().strip().split()
        arr = np.zeros(dimension, dtype=np.float32)

        for word in words:
            # Generate deterministic index between 0 and dimension-1
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dimension
            # Increment frequency-weight mapping
            arr[idx] += 1.0 + (h % 10) / 10.0

        # Inject global sequence hashing context
        seq_hash = int(hashlib.md5(text.lower().encode("utf-8")).hexdigest(), 16)
        for i in range(dimension):
            arr[i] += ((seq_hash + i) % 7) / 20.0

        # Normalize to unit vector
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
            
        return arr.tolist()
