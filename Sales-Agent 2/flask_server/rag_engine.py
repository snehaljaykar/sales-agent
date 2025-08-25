"""
RAG (Retrieval-Augmented Generation) Engine
Handles querying Qdrant and coordinating with LLM services
"""

import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    """Enhanced RAG engine that integrates with existing DataLoader functionality"""
    
    def __init__(self, data_loader):
        """Initialize with existing DataLoader instance"""
        self.data_loader = data_loader
        self.qdrant = data_loader.qdrant
        self.model = data_loader.model
        self.collection_name = data_loader.collection_name
        
        # CSV data access
        self.files_df = data_loader.files_df
        self.users_df = data_loader.users_df  
        self.file_participants_df = data_loader.file_participants_df
    
    def search_transcripts(self, user_text: str, user_id: str, limit: int = 10) -> List[Dict]:

        try:
            return self._semantic_search(user_text, user_id, limit)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _semantic_search(self, search_text: str, user_id: str, limit: int) -> List[Dict]:
        """Semantic vector search"""
        
        if not search_text.strip():
            return []

        # Create query vector
        query_vector = self.model.encode(search_text).tolist()
        
        # Create filter for user's files
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="speakers",
                    match=models.MatchValue(value=user_id)
                )
            ]
        )

        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        
        return [{"id": hit.id, "payload": hit.payload, "score": hit.score} for hit in search_result]
