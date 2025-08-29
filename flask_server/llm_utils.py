import logging
import re
from typing import List, Dict, Any, Optional, Mapping
from enum import Enum

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForLLMRun
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# Global instances
query_converter = None
answer_generator = None


class QueryType(Enum):
    LIST_CALLS = "list_calls"
    SUMMARIZE = "summarize"
    SENTIMENT = "sentiment"
    SEARCH = "search"
    SPECIFIC_CALL = "specific_call"
    COMPLEX = "complex"


class HuggingFaceLLM(LLM):
    """Custom LLM wrapper for HuggingFace models to avoid deprecation warning"""

    pipeline: Any

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""
        response = self.pipeline(prompt, max_new_tokens=256, **kwargs)
        if isinstance(response, list):
            # Get the generated text from the response
            text = response[0]['generated_text']
            # For text-generation pipelines, remove the input prompt from the output
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text
        return str(response)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": str(self.pipeline.model)}


class QueryConverter:
    """Converts natural language queries to structured search parameters"""

    def __init__(self, llm):
        self.llm = llm
    #     self.query_patterns = {
    #         QueryType.LIST_CALLS: [
    #             r"list.*call", r"show.*call", r"what.*calls", r"my calls"
    #         ],
    #         QueryType.SUMMARIZE: [
    #             r"summar", r"recap", r"overview", r"brief"
    #         ],
    #         QueryType.SENTIMENT: [
    #             r"negative", r"concern", r"complaint", r"issue", r"problem"
    #         ],
    #         QueryType.SPECIFIC_CALL: [
    #             r"last call", r"recent call", r"previous call", r"latest call"
    #         ]
    #     }
    #
    # def identify_query_type(self, query: str) -> QueryType:
    #     """Identify the type of query based on patterns"""
    #     query_lower = query.lower()
    #
    #     for query_type, patterns in self.query_patterns.items():
    #         if any(re.search(pattern, query_lower) for pattern in patterns):
    #             return query_type
    #
    #     return QueryType.SEARCH  # Default to general search


class AnswerGenerator:
    """Generates answers using LangChain agent with multi-step reasoning"""

    def __init__(self, llm, rag_engine):
        self.llm = llm
        self.rag_engine = rag_engine

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]], user_name: str) -> str:
        """
        Generate answer using agent or direct response

        Args:
            query: User's query
            retrieved_docs: Documents retrieved from RAG
            user_name: Current user's name

        Returns:
            Generated answer
        """
        try:
            # For complex queries that need multi-step reasoning, use agent
            query_lower = query.lower()

            # Handle common queries directly to avoid agent token limit issues

            # 1. List calls
            if any(phrase in query_lower for phrase in ["list my call", "my call ids", "show my call"]):
                logger.info("Handling list calls query directly")
                call_ids = self.rag_engine.get_user_calls(user_name)
                if call_ids:
                    return f"I found {len(call_ids)} calls for you:\n" + "\n".join(f"- {cid}" for cid in call_ids[:10])
                return "No calls found for your account."

            # 2. Summarize last call
            elif "summarize" in query_lower and "last call" in query_lower:
                logger.info("Handling summarize last call query directly")
                last_call = self.rag_engine.get_last_call(user_name)
                if not last_call:
                    return "No calls found for your account."

                chunks = self.rag_engine.get_call_content(last_call, user_name)
                if not chunks:
                    return f"No content found for call {last_call}"

                # Combine chunks into a manageable context
                content_parts = []
                total_chars = 0
                for chunk in chunks:
                    chunk_content = chunk["content"]
                    if total_chars + len(chunk_content) > 2000:
                        break
                    content_parts.append(chunk_content)
                    total_chars += len(chunk_content)

                content = "\n".join(content_parts)
                prompt = f"Summarize this call transcript:\n\n{content}\n\nSummary:"

                try:
                    response = self.llm(prompt)
                    return f"Summary of {last_call}:\n\n{response.strip()}"
                except Exception as e:
                    logger.error(f"Error generating summary: {e}")
                    return "Error generating summary. The call transcript may be too long."

            # 3. Negative pricing comments
            elif "negative" in query_lower and "pricing" in query_lower:
                logger.info("Handling negative pricing comments directly")
                # Search for pricing-related content
                docs = self.rag_engine.search_transcripts("pricing price cost discount quote budget expensive",
                                                          user_name, limit=20)

                if not docs:
                    return "No discussions about pricing found in your calls."

                # Filter for negative sentiment
                negative_keywords = ['concerned', 'expensive', 'high', 'competitor', 'cheaper', 'issue', 'problem',
                                     'worried', 'budget', 'difficult', 'challenge']
                negative_comments = []

                for doc in docs:
                    content_lower = doc["content"].lower()
                    if any(keyword in content_lower for keyword in negative_keywords):
                        negative_comments.append({
                            'call_id': doc['call_id'],
                            'content': doc['content']
                        })

                if not negative_comments:
                    return "No negative comments about pricing found in your calls."

                # Format response
                response = "Negative comments about pricing:\n\n"
                for i, comment in enumerate(negative_comments[:5], 1):
                    response += f"{i}. From {comment['call_id']}:\n{comment['content'][:200]}...\n\n"

                return response

            # 4. For all other queries, use direct search and response
            else:
                logger.info("Using direct search and response generation")
                # Search for relevant content
                retrieved_docs = self.rag_engine.search_transcripts(query, user_name, limit=5)

                if not retrieved_docs:
                    return "I couldn't find any relevant information in your call transcripts."

                # Generate response from retrieved docs
                context_parts = []
                total_chars = 0
                for doc in retrieved_docs:
                    doc_content = f"[{doc['call_id']}]: {doc['content']}"
                    if total_chars + len(doc_content) > 1500:  # Keep context manageable
                        break
                    context_parts.append(doc_content)
                    total_chars += len(doc_content)

                context = "\n\n".join(context_parts)

                # Simple prompt to stay under token limit
                prompt = f"Answer based on these call excerpts:\n\n{context}\n\nQuestion: {query}\nAnswer:"

                try:
                    response = self.llm(prompt)
                    return response.strip()
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    return "I found relevant information but encountered an error generating the response. Please try a simpler query."

            # if use_agent:
            #     # Use agent for multi-step reasoning
            #     logger.info("Using agent for complex query processing")
            #     try:
            #         # Use invoke instead of run (deprecated)
            #         response = self.agent_executor.invoke({
            #             "input": query,
            #             "user_name": user_name
            #         })
            #         # Extract the output from the response dict
            #         if isinstance(response, dict) and 'output' in response:
            #             return response['output']
            #         else:
            #             return str(response)
            #     except Exception as e:
            #         logger.error(f"Agent executor failed: {e}")
            #         # Fallback to direct response if agent fails
            #         retrieved_docs = self.rag_engine.search_transcripts(query, user_name, limit=5)
            #         if retrieved_docs:
            #             return self._generate_direct_response(query, retrieved_docs, user_name)
            #         return "I encountered an error with the agent. Please try rephrasing your query."
            #
            # else:
            #     # For simple queries, generate direct response from retrieved docs
            #     logger.info("Using direct response generation")
            #     if not retrieved_docs:
            #         # Try searching first
            #         retrieved_docs = self.rag_engine.search_transcripts(query, user_name, limit=5)
            #
            #     if not retrieved_docs:
            #         return "I couldn't find any relevant information in your call transcripts."
            #
            #     # Format context from retrieved documents
            #     context_parts = []
            #     for doc in retrieved_docs[:5]:  # Use top 5 docs
            #         context_parts.append(f"[{doc['call_id']}]: {doc['content']}")
            #
            #     context = "\n\n".join(context_parts)
            #
            #     # Generate response
            #     prompt = f"""Based on the following call transcript excerpts, answer the user's question.
            #
            #         User: {user_name}
            #         Question: {query}
            #
            #         Relevant excerpts:
            #         {context}
            #
            #         Provide a clear, concise answer based on the information above. If the excerpts don't contain enough information, say so.
            #
            #         Answer:"""
            #
            #     response = self.llm(prompt)  # Use __call__ method instead of invoke
            #     return response.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to simple response
            if retrieved_docs:
                return f"Here's what I found: {retrieved_docs[0]['content'][:300]}..."
            return "I encountered an error processing your query. Please try rephrasing it."

    def _generate_direct_response(self, query: str, docs: List[Dict[str, Any]], user_name: str) -> str:
        """Helper method to generate direct response from documents"""
        context_parts = []
        for doc in docs[:5]:  # Use top 5 docs
            context_parts.append(f"[{doc['call_id']}]: {doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate response
        prompt = f"""Based on the following call transcript excerpts, answer the user's question.

        User: {user_name}
        Question: {query}
        
        Relevant excerpts:
        {context}
        
        Provide a clear, concise answer based on the information above. If the excerpts don't contain enough information, say so.
        
        Answer:"""

        response = self.llm(prompt)
        return response.strip()


def initialize_llm_services(
        llm_model: str = "google/flan-t5-small",
        rag_engine=None
):
    """Initialize global LLM service instances"""
    global query_converter, answer_generator

    try:
        logger.info(f"Initializing LLM services with model: {llm_model}")

        # For T5 models, we need to use different pipeline
        if "t5" in llm_model.lower():
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            tokenizer = T5Tokenizer.from_pretrained(llm_model)
            model = T5ForConditionalGeneration.from_pretrained(
                llm_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )

            # Create pipeline for T5
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=256,
                temperature=0.7,
                do_sample=True
            )
        else:
            # Original code for other models
            tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                trust_remote_code=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Create custom LLM wrapper
        llm = HuggingFaceLLM(pipeline=pipe)

        # Initialize services
        query_converter = QueryConverter(llm)

        logger.info("LLM services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize LLM services: {e}")
        raise


def set_rag_engine(rag_engine):
    """Set RAG engine for answer generator (called from app.py after initialization)"""
    global answer_generator, query_converter

    if query_converter and query_converter.llm:
        answer_generator = AnswerGenerator(query_converter.llm, rag_engine)
        logger.info("Answer generator initialized with RAG engine")
    else:
        logger.error("Cannot set RAG engine - LLM services not initialized")