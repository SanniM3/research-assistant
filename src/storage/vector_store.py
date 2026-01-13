"""Vector store for semantic search over chunks."""
from typing import List, Optional, Dict, Any, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

from ..models.chunk import Chunk
from ..config.settings import get_settings


class VectorStore:
    """
    Vector store for semantic search over document chunks.
    
    Uses FAISS for efficient similarity search with OpenAI embeddings.
    Provides methods for adding chunks, searching, and managing the index.
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_path: Optional path to persist/load the index
        """
        settings = get_settings()
        self.persist_path = persist_path or settings.vector_store_path
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        self._store: Optional[FAISS] = None
        self._chunk_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Try to load existing index
        self._load_or_create()
    
    def _load_or_create(self) -> None:
        """Load existing index or create new one."""
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                self._store = FAISS.load_local(
                    self.persist_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception:
                self._store = None
    
    def add_chunks(self, chunks: List[Chunk]) -> List[str]:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
            
        Returns:
            List of chunk IDs that were added
        """
        if not chunks:
            return []
        
        # Convert chunks to documents
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "section_path": chunk.section_path,
                    "source_type": chunk.source_type.value,
                    "quality": chunk.quality.value,
                }
            )
            documents.append(doc)
            self._chunk_metadata[chunk.chunk_id] = chunk.model_dump()
        
        # Add to vector store
        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            self._store.add_documents(documents)
        
        return [chunk.chunk_id for chunk in chunks]
    
    def search(self, query: str, k: int = 5, 
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of (Chunk, score) tuples
        """
        if self._store is None:
            return []
        
        # Perform similarity search
        results = self._store.similarity_search_with_score(query, k=k * 2)
        
        # Filter and convert to Chunks
        filtered_results = []
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id in self._chunk_metadata:
                # Apply filters if specified
                if filters:
                    chunk_meta = self._chunk_metadata[chunk_id]
                    if not all(chunk_meta.get(fk) == fv for fk, fv in filters.items()):
                        continue
                
                chunk = Chunk(**self._chunk_metadata[chunk_id])
                filtered_results.append((chunk, score))
                
                if len(filtered_results) >= k:
                    break
        
        return filtered_results
    
    def search_by_paper(self, paper_id: str, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search within a specific paper's chunks."""
        return self.search(query, k=k, filters={"paper_id": paper_id})
    
    def get_chunks_for_paper(self, paper_id: str) -> List[Chunk]:
        """Get all chunks for a specific paper."""
        chunks = []
        for chunk_id, metadata in self._chunk_metadata.items():
            if metadata.get("paper_id") == paper_id:
                chunks.append(Chunk(**metadata))
        return chunks
    
    def delete_paper_chunks(self, paper_id: str) -> int:
        """Delete all chunks for a paper. Returns count of deleted chunks."""
        # Note: FAISS doesn't support deletion well, so we track metadata separately
        deleted = 0
        to_delete = []
        for chunk_id, metadata in self._chunk_metadata.items():
            if metadata.get("paper_id") == paper_id:
                to_delete.append(chunk_id)
                deleted += 1
        
        for chunk_id in to_delete:
            del self._chunk_metadata[chunk_id]
        
        return deleted
    
    def save(self) -> None:
        """Persist the vector store to disk."""
        if self._store and self.persist_path:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            self._store.save_local(self.persist_path)
    
    def count(self) -> int:
        """Get total number of chunks."""
        return len(self._chunk_metadata)
    
    def clear(self) -> None:
        """Clear the vector store."""
        self._store = None
        self._chunk_metadata.clear()
