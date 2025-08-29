import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG Engine for searching and retrieving call transcripts from Qdrant"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.qdrant = data_loader.qdrant
        self.model = data_loader.model
        self.collection_name = data_loader.collection_name

    def search_transcripts(self, query: str, user_name: str, limit: int = 10, call_id: Optional[str] = None) -> List[
        Dict[str, Any]]:
        """
        Search for relevant transcript chunks based on query

        Args:
            query: Search query text
            user_name: User name to filter results
            limit: Maximum number of results to return
            call_id: Optional specific call ID to search within

        Returns:
            List of relevant documents with metadata
        """
        try:
            # Generate embedding for the query
            query_vector = self.model.encode(query).tolist()

            # Build filter conditions
            filter_conditions = []

            # Filter by speakers (user_name) - FIXED: wrap in list
            if user_name:
                filter_conditions.append(
                    models.FieldCondition(
                        key="speakers",
                        match=models.MatchAny(any=[user_name])  # Must be a list
                    )
                )

            # Filter by call_id if provided
            if call_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="call_id",
                        match=models.MatchValue(value=call_id)
                    )
                )

            # Create filter if conditions exist
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(must=filter_conditions)

            # Search Qdrant
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter
            )

            # Format results
            documents = []
            for hit in search_result:
                doc = {
                    "content": hit.payload.get("text_data", ""),
                    "call_id": hit.payload.get("call_id", ""),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "timestamp": hit.payload.get("timestamp", ""),
                    "speakers": hit.payload.get("speakers", []),
                    "score": hit.score
                }
                documents.append(doc)

            logger.info(f"Found {len(documents)} documents for query: {query[:50]}... with user: {user_name}")
            return documents

        except Exception as e:
            logger.error(f"Error searching transcripts: {e}")
            return []

    def get_user_calls(self, user_name: str) -> List[str]:
        """
        Get all unique call IDs for a specific user

        Args:
            user_name: User name to search for

        Returns:
            List of unique call IDs
        """
        try:
            call_ids = set()
            offset = None

            # Filter by user in speakers - FIXED: wrap in list
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="speakers",
                        match=models.MatchAny(any=[user_name])  # Must be a list
                    )
                ]
            )

            # Scroll through all results
            while True:
                result = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=100,
                    offset=offset
                )

                for point in result[0]:
                    call_id = point.payload.get("call_id")
                    if call_id:
                        call_ids.add(call_id)

                offset = result[1]
                if offset is None:
                    break

            logger.info(f"Found {len(call_ids)} calls for user: {user_name}")
            return sorted(list(call_ids))

        except Exception as e:
            logger.error(f"Error getting user calls: {e}")
            return []

    def get_last_call(self, user_name: str) -> Optional[str]:
        """
        Get the most recent call ID for a user

        Args:
            user_name: User name

        Returns:
            Most recent call ID or None
        """
        call_ids = self.get_user_calls(user_name)
        if call_ids:
            # Assuming call IDs are sortable (e.g., by timestamp in filename)
            return call_ids[-1]
        return None

    def get_call_content(self, call_id: str, user_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific call, sorted by chunk index

        Args:
            call_id: Call ID to retrieve
            user_name: Optional user name for additional filtering

        Returns:
            List of chunks sorted by chunk_index
        """
        try:
            chunks = []
            offset = None

            # Build filter
            filter_conditions = [
                models.FieldCondition(
                    key="call_id",
                    match=models.MatchValue(value=call_id)
                )
            ]

            if user_name:
                filter_conditions.append(
                    models.FieldCondition(
                        key="speakers",
                        match=models.MatchAny(any=[user_name])  # Must be a list
                    )
                )

            filter_condition = models.Filter(must=filter_conditions)

            # Scroll through all chunks
            while True:
                result = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=100,
                    offset=offset
                )

                for point in result[0]:
                    chunk = {
                        "content": point.payload.get("text_data", ""),
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "timestamp": point.payload.get("timestamp", ""),
                        "speakers": point.payload.get("speakers", [])
                    }
                    chunks.append(chunk)

                offset = result[1]
                if offset is None:
                    break

            # Sort by chunk index
            chunks.sort(key=lambda x: x["chunk_index"])
            return chunks

        except Exception as e:
            logger.error(f"Error getting call content: {e}")
            return []
