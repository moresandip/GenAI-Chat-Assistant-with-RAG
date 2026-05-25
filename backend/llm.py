import os
import logging
from typing import Dict, Any, Tuple
import google.generativeai as genai
from openai import OpenAI
import openai

logger = logging.getLogger(__name__)

class LLMServiceException(Exception):
    """Custom exception class for structured LLM errors."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

class LLMService:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if self.provider == "gemini":
            if self.gemini_key:
                genai.configure(api_key=self.gemini_key)
                logger.info("Gemini LLM service initialized.")
            else:
                logger.warning("Gemini API key is missing. LLM will run in offline Mock mode.")
                self.provider = "mock"
        elif self.provider == "openai":
            if self.openai_key:
                self.openai_client = OpenAI(api_key=self.openai_key)
                logger.info("OpenAI LLM service initialized.")
            else:
                logger.warning("OpenAI API key is missing. LLM will run in offline Mock mode.")
                self.provider = "mock"
        else:
            logger.info("LLM service initialized in Mock mode.")

    def generate_response(self, prompt: str) -> Tuple[str, int]:
        """
        Sends the compiled prompt to the selected LLM provider.
        Returns a tuple: (response_text, tokens_used)
        """
        if self.provider == "gemini":
            try:
                # Use gemini-2.5-flash as the standard fast LLM
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,  # Grounded answers (req between 0 and 0.3)
                    )
                )
                
                # Check if token usage counts are returned
                tokens = 0
                try:
                    # Request token usage count
                    tokens = model.count_tokens(prompt).total_tokens
                except Exception:
                    pass
                
                return response.text, tokens

            except Exception as e:
                # Catch auth, rate limit, or timeout problems
                err_msg = str(e)
                logger.error(f"Gemini LLM API failed: {err_msg}")
                if "API_KEY_INVALID" in err_msg or "401" in err_msg:
                    raise LLMServiceException("Invalid Gemini API key", status_code=401)
                elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    raise LLMServiceException("Gemini API rate limit exceeded", status_code=429)
                elif "deadline exceeded" in err_msg.lower() or "504" in err_msg:
                    raise LLMServiceException("Gemini API request timed out", status_code=504)
                else:
                    raise LLMServiceException(f"Gemini LLM error: {err_msg}", status_code=502)

        elif self.provider == "openai":
            try:
                # Use gpt-3.5-turbo (fast, cheap) or gpt-4o-mini
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    timeout=15.0  # Timeout handling
                )
                
                tokens = response.usage.total_tokens if response.usage else 0
                return response.choices[0].message.content, tokens

            except openai.AuthenticationError as e:
                logger.error(f"OpenAI Auth failure: {e}")
                raise LLMServiceException("Invalid OpenAI API key", status_code=401)
            except openai.RateLimitError as e:
                logger.error(f"OpenAI RateLimit failure: {e}")
                raise LLMServiceException("OpenAI API rate limit exceeded", status_code=429)
            except openai.APITimeoutError as e:
                logger.error(f"OpenAI Timeout: {e}")
                raise LLMServiceException("OpenAI API request timed out", status_code=504)
            except Exception as e:
                logger.error(f"OpenAI general failure: {e}")
                raise LLMServiceException(f"OpenAI LLM error: {str(e)}", status_code=502)

        else:
            # Local Mock implementation for offline validation
            return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> Tuple[str, int]:
        """
        Synthesizes a response from the context inside the prompt to allow
        offline RAG testing and demonstration.
        """
        logger.info("Synthesizing local offline response from prompt structure.")
        
        # Simple heuristic to extract the retrieved context and question
        context_section = ""
        question_section = ""
        
        if "Context:" in prompt:
            parts = prompt.split("Context:")
            if len(parts) > 1:
                context_part = parts[1].split("Conversation History:")
                context_section = context_part[0].strip()
                
                if "Question:" in prompt:
                    q_part = prompt.split("Question:")
                    if len(q_part) > 1:
                        question_section = q_part[1].split("Answer:")[0].strip()

        # Generate a response based on keywords matched in the retrieved context
        if not context_section or "No relevant context found" in context_section:
            reply = "I could not find enough information in the knowledge base to answer this question."
        else:
            reply = f"**[Offline Mode Mock Response]**\nBased on our internal documents:\n\n"
            # Extract bullet lines or key points from the mock context
            lines = context_section.split("\n")
            useful_facts = []
            for line in lines:
                if line.strip() and not line.startswith("- Chunk") and not line.startswith("Source:"):
                    useful_facts.append(line.strip())
            
            if useful_facts:
                reply += " " + "\n\n".join(useful_facts[:2])
                reply += "\n\n*(Note: This response was generated locally in offline mock mode since no API keys were configured).* "
            else:
                reply = "I could not find enough information in the knowledge base to answer this question."

        # Simulate a token count (approx. 4 characters per token)
        tokens_simulated = len(prompt) // 4
        return reply, tokens_simulated
