# -*- coding: utf-8 -*-
"""
Optimized vector memory builder with intelligent caching and robust filtering.

This module provides comprehensive document processing, vector storage, and
retrieval functionality for the Medical RAG system with advanced caching,
filtering, and performance optimizations.

Features:
- Hierarchical document chunking with parent-child relationships
- Intelligent caching with SQLite persistence
- Robust similarity search with post-filtering
- Document type management (permanent/temporary)
- Performance monitoring and optimization
- Multi-format document support

Author: Souleiman & Abdelbar Medical RAG System
Created: 2025
"""

import os
import re
import json
import platform
import hashlib
import sqlite3
import pickle
from time import sleep
from datetime import datetime, timedelta
from tqdm import tqdm
import unicodedata
from typing import Dict, List, Optional, Any, Tuple

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.docstore.document import Document as LangchainDocument
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from grader import retrieval_grader
from model_settings import Model_Settings
from debugging import get_debugger, debug_decorator

# Initialize debugger
debugger = get_debugger()

# Global settings instance for compatibility
_global_settings = Model_Settings()


def extract_headings_from_page(text: str) -> list[str]:
    """
    Parse a page of text and extract hierarchical headings.
    
    Args:
        text: Text content to parse
        
    Returns:
        List of extracted headings
    """
    headings = []
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Numbered headings (1., 1.1., etc.)
        if re.match(r'^\d+(\.\d+)*\s+\S', line):
            headings.append(line)
        # All caps headings
        elif (len(line) >= 3 and line.upper() == line and
              re.match(r'^[A-ZÀ-Ÿ][A-Z\sÉÈÊÀÂÔÙÇ\-]+$', line)):
            headings.append(line)
            
    return headings


def normalize_text(text: str) -> str:
    """
    Convert Unicode text to compatibility decomposition form.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text string
    """
    return unicodedata.normalize("NFKC", text)


class MemoryBuilder:
    """
    Optimized vector memory manager with intelligent caching and advanced filtering.
    
    Manages document processing, vector storage, and retrieval with comprehensive
    caching, performance monitoring, and robust error handling.
    """

    def __init__(
        self,
        chunk_size: int = 2048,
        chunk_overlap: int = 100,
        parent_chunk_size: int = 4096,
        parent_chunk_overlap: int = 200,
        embedding_model_name: str = 'mxbai-embed-large',
        vector_store_dir: str = "chroma_vectorstore",
        collection_name: str = "MedBot",
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
        settings: Model_Settings = None
    ):
        """
        Initialize the memory builder.
        
        Args:
            chunk_size: Size of child chunks
            chunk_overlap: Overlap between child chunks
            parent_chunk_size: Size of parent chunks
            parent_chunk_overlap: Overlap between parent chunks
            embedding_model_name: Name of embedding model
            vector_store_dir: Directory for vector store
            collection_name: Name of the collection
            cache_enabled: Whether to enable caching
            cache_ttl_hours: Cache time-to-live in hours
            settings: Model settings instance
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = cache_ttl_hours

        # Use provided settings or global instance
        self.settings = settings or _global_settings

        # Memory and persistent cache
        self.memory_cache = {}
        os.makedirs("system_files/cache", exist_ok=True)
        self.cache_db = "system_files/cache/query_cache.db"
        self.sources = []

        if cache_enabled:
            self._init_cache_db()
            
        debugger.log_info(
            "Constructeur de mémoire vectorielle initialisé",
            "Vector memory builder initialized"
        )

    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        conn = sqlite3.connect(self.cache_db)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                results BLOB,
                timestamp REAL,
                hits INTEGER DEFAULT 1
            )
        ''')
        conn.commit()
        conn.close()
        
        debugger.log_info("Base de données de cache initialisée", "Cache database initialized")

    @debug_decorator(debugger, "instantiate_memory_builder",
                    "Configuration des splitters et modèles",
                    "Configure splitters and models")
    def instantiate(self):
        """Configure splitters, embedding model, and database connections."""
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            length_function=len,
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
        )

        self.embedding_model = OllamaEmbeddings(model=self.embedding_model_name)

        os.makedirs(self.vector_store_dir, exist_ok=True)

        self.vectorstore = Chroma(
            persist_directory=self.vector_store_dir,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

        self.store = InMemoryStore()

        self.chroma_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        self.sources = []

    def _get_cache_key(self, question: str, k: int, threshold: float, where: dict = None) -> str:
        """
        Generate cache key based on query parameters.
        
        Args:
            question: Query question
            k: Number of results
            threshold: Similarity threshold
            where: Filter conditions
            
        Returns:
            MD5 hash string for cache key
        """
        cache_data = f"{question}_{k}_{threshold}_{json.dumps(where, sort_keys=True) if where else 'None'}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[List]:
        """
        Retrieve cached result if valid.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached result list or None
        """
        if not self.cache_enabled:
            return None

        # Memory cache first
        if cache_key in self.memory_cache:
            cached_data, timestamp = self.memory_cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl_hours * 3600:
                return cached_data
            else:
                del self.memory_cache[cache_key]

        # Persistent cache
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT results, timestamp, hits FROM query_cache WHERE query_hash = ?",
                (cache_key,)
            )

            result = cursor.fetchone()
            if result:
                cached_results, timestamp, hits = result
                
                # Check TTL
                if datetime.now().timestamp() - timestamp < self.cache_ttl_hours * 3600:
                    # Increment hit counter
                    cursor.execute(
                        "UPDATE query_cache SET hits = hits + 1 WHERE query_hash = ?",
                        (cache_key,)
                    )
                    conn.commit()
                    conn.close()

                    # Deserialize and add to memory cache
                    results = pickle.loads(cached_results)
                    self.memory_cache[cache_key] = (results, timestamp)
                    return results
                else:
                    # Remove expired cache
                    cursor.execute("DELETE FROM query_cache WHERE query_hash = ?", (cache_key,))
                    conn.commit()
                    
            conn.close()
        except Exception as e:
            debugger.log_error("Erreur de lecture du cache", "Cache read error", e)

        return None

    def _cache_result(self, cache_key: str, results: List):
        """
        Cache a result.
        
        Args:
            cache_key: Key to store under
            results: Results to cache
        """
        if not self.cache_enabled:
            return

        timestamp = datetime.now().timestamp()

        # Memory cache
        self.memory_cache[cache_key] = (results, timestamp)

        # Persistent cache
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()

            # Serialize results
            serialized_results = pickle.dumps(results)

            cursor.execute(
                "INSERT OR REPLACE INTO query_cache (query_hash, results, timestamp, hits) VALUES (?, ?, ?, 1)",
                (cache_key, serialized_results, timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            debugger.log_error("Erreur d'écriture du cache", "Cache write error", e)

    def clean_expired_cache(self):
        """Clean expired cache entries."""
        if not self.cache_enabled:
            return

        cutoff_time = datetime.now().timestamp() - (self.cache_ttl_hours * 3600)

        # Clean memory cache
        expired_keys = [
            key for key, (_, timestamp) in self.memory_cache.items()
            if timestamp < cutoff_time
        ]

        for key in expired_keys:
            del self.memory_cache[key]

        # Clean persistent cache
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM query_cache WHERE timestamp < ?", (cutoff_time,))
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted_count > 0:
                debugger.log_info(f"Nettoyé {deleted_count} entrées de cache expirées",
                                f"Cleaned {deleted_count} expired cache entries")
        except Exception as e:
            debugger.log_error("Erreur de nettoyage du cache", "Cache cleanup error", e)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        if not self.cache_enabled:
            return {"cache_enabled": False}

        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(hits), SUM(hits) FROM query_cache")
            total_entries, avg_hits, total_hits = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) FROM query_cache WHERE timestamp > ?",
                          (datetime.now().timestamp() - 3600,))  # Last hour
            recent_entries = cursor.fetchone()[0]
            conn.close()

            return {
                "cache_enabled": True,
                "total_entries": total_entries or 0,
                "memory_cache_entries": len(self.memory_cache),
                "average_hits_per_entry": round(avg_hits or 0, 2),
                "total_cache_hits": total_hits or 0,
                "entries_last_hour": recent_entries,
                "cache_ttl_hours": self.cache_ttl_hours
            }

        except Exception as e:
            debugger.log_error("Erreur de récupération des statistiques de cache",
                              "Error retrieving cache statistics", e)
            return {"cache_enabled": True, "error": str(e)}

    @debug_decorator(debugger, "similarity_search_with_score",
                    "Recherche de similarité optimisée avec cache",
                    "Optimized similarity search with cache")
    def vectorstore_similarity_search_with_score(
        self,
        question: str,
        k: int,
        retrieval_threshold: float,
        where: dict = None,
        debug: bool = False
    ) -> List[Tuple[LangchainDocument, float]]:
        """
        Optimized similarity search with caching and advanced filtering.
        
        Args:
            question: Search question
            k: Number of results to return
            retrieval_threshold: Similarity threshold
            where: Filter conditions
            debug: Enable debug logging
            
        Returns:
            List of (document, score) tuples
        """
        # Generate cache key
        cache_key = self._get_cache_key(question, k, retrieval_threshold, where)

        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            if debug:
                debugger.log_info("Utilisation du résultat en cache", "Using cached result")
            return cached_result

        # Perform search if not in cache
        results = self._perform_similarity_search(question, k, retrieval_threshold, where, debug)

        # Cache the result
        self._cache_result(cache_key, results)

        return results

    @debug_decorator(debugger, "perform_similarity_search",
                    "Recherche de similarité avec filtrage robuste",
                    "Similarity search with robust filtering")
    # FIXED APPROACH - Use native filtering
    def _perform_similarity_search(self, question, k, retrieval_threshold, where=None, debug=False):
        """FIXED: Use ChromaDB native filtering instead of post-processing."""
        try:
            # KEY FIX: Pass filter directly to ChromaDB
            raw_results = self.vectorstore.similarity_search_with_score(
                query=question, 
                k=k,
                filter=where  # Native filtering at DB level
            )
            
            if debug:
                debugger.log_debug(f"ChromaDB returned {len(raw_results)} results", 
                                f"ChromaDB returned {len(raw_results)} results")
        except Exception as e:
            debugger.log_error("ChromaDB search error", "ChromaDB search error", e)
            return []

        # Apply grader filtering if enabled
        if self.settings.IS_GRADER and raw_results:
            graded_results = []
            for doc, score in raw_results:
                try:
                    grader_result = retrieval_grader(question, doc.page_content)
                    if grader_result.get('score', 0) == 1:
                        graded_results.append((doc, score))
                except:
                    graded_results.append((doc, score))
            raw_results = graded_results

        # Apply threshold
        return [(doc, score) for doc, score in raw_results if score >= retrieval_threshold]


    # All other methods identical to your version (list_pdf_files, pdf_file_reader, etc.)

    def list_pdf_files(self, folder_path: str = 'uploaded_docs') -> list[str]:
        """
        List PDF files in directory.
        
        Args:
            folder_path: Path to search for PDFs
            
        Returns:
            List of PDF file paths
        """
        if not os.path.exists(folder_path):
            return []

        return [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith('.pdf')
        ]

    def pdf_file_reader(self, file_path: str) -> list[LangchainDocument]:
        """
        Read and split PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of document chunks
        """
        loader = PyPDFLoader(file_path=file_path, extract_images=False, headers=None)
        return loader.load_and_split()

    def text_file_reader(self, file_path: str) -> str:
        """
        Read text file content.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding='utf8') as f:
            return f.read()

    def _get_doc_type(self, path: str) -> str:
        """
        Determine document type from path.
        
        Args:
            path: File path to analyze
            
        Returns:
            Document type string
        """
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) >= 2:
            subfolder = parts[-2]
            if subfolder in ['permanent', 'temporary']:
                return subfolder
        return "unknown"

    @debug_decorator(debugger, "add_document_to_vectorstore",
                    "Ajout d'un document au store vectoriel",
                    "Add document to vector store")
    def vectorstore_add_document(self, text: str, source: str, page_num: int = None, doc_type: str = "unknown"):
        """
        Add document to vector store.
        
        Args:
            text: Document text content
            source: Document source name
            page_num: Optional page number
            doc_type: Document type classification
        """
        clean = normalize_text(text)
        headings = extract_headings_from_page(clean)

        doc = LangchainDocument(
            page_content=clean,
            metadata={
                "source": source,
                "page": page_num or "N/A",
                "headings": json.dumps(headings),
                "date": str(datetime.now()),
                "doc_type": doc_type,
            },
        )

        splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=16000, chunk_overlap=300)
        chunks = splitter.split_documents([doc])

        for chunk in chunks:
            chunk.metadata.update({
                "page": page_num or "N/A",
                "headings": json.dumps(headings),
                "doc_type": doc_type,
            })

        if chunks:
            self.chroma_retriever.add_documents(chunks, ids=None)

    def _get_file_name(self, file_path: str) -> str:
        """
        Extract filename from path.
        
        Args:
            file_path: Full file path
            
        Returns:
            Filename only
        """
        sep = "\\" if platform.system() == "Windows" else "/"
        return file_path.split(sep)[-1]

    @debug_decorator(debugger, "add_multiple_files",
                    "Ajout de plusieurs fichiers au store vectoriel",
                    "Add multiple files to vector store")
    def vectorstore_add_multi_files(self, path_files: list[str]) -> str:
        """
        Add multiple files to vector store.
        
        Args:
            path_files: List of file paths to add
            
        Returns:
            Summary string of added files
        """
        summary = ""
        self.list_sources_in_db()

        for idx, path in enumerate(path_files, start=1):
            name = self._get_file_name(path)
            ext = name.lower().split('.')[-1]
            doc_type = self._get_doc_type(path)

            if name in self.sources:
                continue

            summary += f"({idx}/{len(path_files)}) {name} [{doc_type}]\n"

            if ext == "pdf":
                pages = self.pdf_file_reader(path)
                for i, page in enumerate(tqdm(pages, desc="Indexing PDF")):
                    if page.page_content:
                        self.vectorstore_add_document(page.page_content, name, page_num=i+1, doc_type=doc_type)
                    sleep(0.05)  # Reduced delay for performance
                    
            elif ext in ["txt", "md", "mdx"]:
                text = self.text_file_reader(path)
                if text:
                    self.vectorstore_add_document(text, name, doc_type=doc_type)
            else:
                summary += f" Unsupported file type: {name}\n"

        # Clean cache after adding new documents
        if self.cache_enabled:
            self.memory_cache.clear()
            debugger.log_info("Cache vidé après mise à jour des documents", "Cache cleared after document update")

        self.list_sources_in_db()
        return summary

    def list_sources_in_db(self, repr: bool = False) -> list[str]:
        """
        List all sources in database.
        
        Args:
            repr: Whether to print sources
            
        Returns:
            List of source names
        """
        try:
            metas = self.vectorstore._collection.get(include=["metadatas"])["metadatas"]
            self.sources = list({meta["source"] for meta in metas if "source" in meta})
        except Exception as e:
            debugger.log_warning("Could not list sources", "Could not list sources")
            self.sources = []

        if repr:
            for src in self.sources:
                print(src)

        return self.sources

    def list_sources_by_type(self, doc_type: str) -> list[str]:
        """
        List sources by document type.
        
        Args:
            doc_type: Type to filter by
            
        Returns:
            List of matching source names
        """
        try:
            metas = self.vectorstore._collection.get(include=["metadatas"])["metadatas"]
            filtered_sources = list({
                meta["source"] for meta in metas
                if meta.get("doc_type") == doc_type and "source" in meta
            })
            return filtered_sources
        except Exception as e:
            debugger.log_warning("Could not list sources by type", "Could not list sources by type")
            return []

    def delete_source(self, source: str):
        """
        Delete source from database.
        
        Args:
            source: Source name to delete
        """
        self.vectorstore._collection.delete(where={"source": {"$eq": source}})
        
        # Clean cache after deletion
        if self.cache_enabled:
            self.memory_cache.clear()
        
        self.list_sources_in_db()
        debugger.log_info(f"Source supprimée: {source}", f"Source deleted: {source}")

    def clear(self):
        """Clear all documents from database."""
        try:
            collection_data = self.vectorstore._collection.get()
            if collection_data and collection_data.get("ids"):
                self.vectorstore._collection.delete(ids=collection_data["ids"])
                
            self.store.mdelete(list(self.store.yield_keys()))
            self.sources = []

            # Clean cache
            if self.cache_enabled:
                self.memory_cache.clear()
                try:
                    conn = sqlite3.connect(self.cache_db)
                    conn.execute("DELETE FROM query_cache")
                    conn.commit()
                    conn.close()
                except:
                    pass

            debugger.log_info("Toutes les sources supprimées de la base de données", "All sources removed from database")
        except Exception as e:
            debugger.log_error("Erreur lors du vidage de la base de données", "Error clearing database", e)

    def clear_temporary(self):
        """Clear temporary documents from database."""
        try:
            self.vectorstore._collection.delete(where={"doc_type": {"$eq": "temporary"}})
            self.store.mdelete(list(self.store.yield_keys()))

            # Clean cache after modification
            if self.cache_enabled:
                self.memory_cache.clear()

            self.list_sources_in_db()
            debugger.log_info("Documents temporaires supprimés de la base de données", "Temporary documents removed from database")
        except Exception as e:
            debugger.log_error("Erreur lors du nettoyage des documents temporaires", "Error clearing temporary documents", e)


def main():
    """Entrypoint placeholder; actual orchestration lives in RAGSystem or main application."""
    pass


if __name__ == "__main__":
    main()
    