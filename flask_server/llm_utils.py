"""
LLM utilities for query conversion and answer generation
Uses local LLM models for intelligent processing
"""

import json
from typing import List, Dict, Any
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalLLMManager:
    """Manages local LLM instances for query conversion and answer generation"""

    def __init__(self):
        """
        Initialize LLM manager with tinyllama 4-bit
        """
        self.model_name = "google/FLAN-T5-Large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load text generation pipeline"""
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )
            logger.info(f"LLM pipeline loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load LLM pipeline: {e}")
            self.pipeline = None

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, return_tensors="pt")[0]

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class QueryConverter:
    """Converts natural language queries to Qdrant search parameters"""

    def __init__(self, llm_manager: LocalLLMManager):
        self.llm = llm_manager

    def convert_to_search_query(self, user_query: str, user_id: str) -> Dict[str, Any]:
        # You can implement LLM-based conversion or fallback rules
        # Currently returns empty dict
        return self._user_based_conversion(user_query, user_id)

    def _user_based_conversion(self, user_query: str, user_id: str) -> Dict[str, Any]:
        """Use LLM to convert query to search parameters (placeholder)"""
        prompt = f"""Convert this user query into search parameters for a call transcript database.

                        User Query: "{user_query}" with user_id: {user_id}
                        
                        You are a Qdrant filter generator.
                        Available metadata fields:
                        - call_id (string): unique call identifier
                        - speakers (list of strings): participants in the call
                        - chunk_index (integer): index of transcript chunk
                        - timestamp (datetime string or number): when the call happened
                        
                        Rules:
                        - Always output a valid Qdrant filter JSON.
                        - Use `must` clause for strict matching.
                        - If the query is "my last call", filter by speaker=user_id and sort by timestamp DESC.
                        - Output ONLY the filter JSON (no explanation)."""
        try:
            if self.llm.pipeline:
                response = self.llm.pipeline(
                    prompt,
                    max_length=256,
                    return_full_text=False,
                    pad_token_id=self.llm.tokenizer.eos_token_id
                )
                return json.loads(response[0]["generated_text"].strip())
            else:
                return {}  # fallback
        except Exception as e:
            logger.warning(f"LLM conversion failed: {e}. Using fallback.")
            return {}


class AnswerGenerator:
    """Generates answers based on retrieved documents and user queries"""

    def __init__(self, llm_manager: LocalLLMManager, max_chunk_tokens: int = 512):
        self.llm = llm_manager
        self.max_chunk_tokens = max_chunk_tokens  # chunk size for CPU long inputs

    def generate_answer(self, user_query: str, retrieved_docs: List[Dict], search_params: Dict) -> str:
        """Generate answer based on query and retrieved documents"""
        if not retrieved_docs:
            return "I couldn't find any relevant information in your call transcripts for this query."
        if self.llm.pipeline is not None:
            return self._llm_based_generation(user_query, retrieved_docs, search_params)
        else:
            return "No model available for generation"

    def _llm_based_generation(self, user_query: str, retrieved_docs: List[Dict], search_params: Dict) -> str:
        """Use LLM to generate contextual answer, chunking long contexts"""

        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            payload = doc.get("payload", {})
            filename = payload.get("call_id", "Unknown")
            text = payload.get("text_data", "")
            timestamp = payload.get("timestamp", "")
            context_parts.append(f"Call {filename}_{timestamp}: {text}")

        # Tokenize full context
        all_tokens = self.llm.tokenizer.encode("; ".join(context_parts), add_special_tokens=False)
        logger.info(f"Total tokens in context: {len(all_tokens)}")

        # Chunk tokens for CPU
        chunks = [
            all_tokens[i:i + self.max_chunk_tokens]
            for i in range(0, len(all_tokens), self.max_chunk_tokens)
        ]

        chunk_summaries = []
        for i, chunk_tokens in enumerate(chunks):
            print(f"Processing chunk {i+1} out of {len(chunks)}")
            chunk_text = self.llm.detokenize(chunk_tokens)
            prompt = f"""
                        You are an AI assistant that answers questions strictly based on the context provided.
                        Use only the information present. If the answer is not in the context, respond with 'Answer not found in the documents'.
                        
                        User Query: {user_query}
                        
                        Context:
                        {chunk_text}
                        
                        Answer:
                        """
            try:
                # Calculate max_length safely
                input_len = len(self.llm.tokenizer.encode(prompt, add_special_tokens=False))
                max_length = input_len + 100  # allow 100 tokens generation
                response = self.llm.pipeline(
                    prompt,
                    max_length=max_length,
                    return_full_text=False,
                    pad_token_id=self.llm.tokenizer.eos_token_id
                )
                answer_chunk = response[0]["generated_text"].strip()
                chunk_summaries.append(answer_chunk)
            except Exception as e:
                logger.warning(f"LLM generation failed for a chunk: {e}")
                chunk_summaries.append("Answer not found in the documents.")

        # Combine chunk summaries
        final_answer = " ".join(chunk_summaries)
        return final_answer


# Global instances (initialized in app.py)
llm_manager = None
query_converter = None
answer_generator = None


def initialize_llm_services():
    """Initialize global LLM services"""
    global llm_manager, query_converter, answer_generator

    logger.info("Initializing LLM services...")
    llm_manager = LocalLLMManager()
    query_converter = QueryConverter(llm_manager)
    answer_generator = AnswerGenerator(llm_manager)
    logger.info("LLM services initialized successfully!")
