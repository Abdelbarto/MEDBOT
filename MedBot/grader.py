# -*- coding: utf-8 -*-
"""
Optimized retrieval grader with caching and performance metrics.

This module provides document relevance evaluation for the Medical RAG system
with intelligent caching, performance monitoring, and robust error handling.

Features:
- LLM-based relevance grading
- Intelligent caching with TTL
- Performance metrics tracking
- Robust error handling
- Compatible interface with existing system

Author: Medical RAG System
Created: 2025
"""

from typing import Dict, Any, Optional, List
import json
import time
import hashlib
import os
from functools import lru_cache
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from model_settings import Model_Settings
from debugging import get_debugger, debug_decorator

# Initialize debugger
debugger = get_debugger()


class RetrievalGrader:
    """
    Optimized relevance evaluator with caching and performance metrics.
    
    Evaluates document relevance for medical queries with intelligent caching
    and detailed performance tracking.
    """

    def __init__(self, cache_size: int = 1000, cache_ttl_minutes: int = 60):
        """
        Initialize the retrieval grader.
        
        Args:
            cache_size: Maximum cache entries to keep
            cache_ttl_minutes: Time to live for cache entries in minutes
        """
        self.cache_size = cache_size
        self.cache_ttl_minutes = cache_ttl_minutes
        
        # Create cache directory if it doesn't exist
        self.cache_dir = "system_files/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Grading statistics
        self.grading_stats = {
            "total_grades": 0,
            "cache_hits": 0,
            "positive_grades": 0,
            "negative_grades": 0,
            "errors": 0,
            "avg_time": 0.0,
            "total_time": 0.0
        }

        # Cache with timestamp for TTL
        self._cache = {}
        
        # LLM grader initialization (lazy loading)
        self._grader_llm = None
        self._grader_chain = None
        
        debugger.log_info(
            "Grader de pertinence initialisé",
            "Retrieval grader initialized"
        )

    def _get_cache_key(self, question: str, document: str) -> str:
        """
        Generate cache key based on content hash.
        
        Args:
            question: Question text
            document: Document content
            
        Returns:
            MD5 hash string for cache key
        """
        content = f"{question}||{document[:1000]}"  # Limit for performance
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """
        Check if cache entry is still valid.
        
        Args:
            timestamp: Entry timestamp
            
        Returns:
            True if cache entry is valid
        """
        return (datetime.now().timestamp() - timestamp) < (self.cache_ttl_minutes * 60)

    @debug_decorator(debugger, "initialize_grader",
                    "Initialisation du LLM grader",
                    "LLM grader initialization")
    def _initialize_grader(self, model_name: Optional[str] = None):
        """
        Initialize LLM grader with error handling.
        
        Args:
            model_name: Optional model name to use
        """
        if self._grader_llm is not None:
            return

        try:
            settings = Model_Settings()
            model_to_use = model_name or settings.MODEL_NAME

            # Optimized prompt for more accurate evaluation
            prompt_template = """
Vous êtes un évaluateur expert de la pertinence de documents médicaux.

Évaluez si le DOCUMENT fourni contient des informations pertinentes pour répondre à la QUESTION.

CRITÈRES D'ÉVALUATION :
- Le document doit contenir des informations directement liées à la question
- Les informations doivent être médicalement pertinentes et exploitables
- Une pertinence partielle compte comme pertinente (score: 1)
- Seuls les documents complètement hors-sujet reçoivent un score de 0

QUESTION : {question}

DOCUMENT : {documents}

Répondez UNIQUEMENT par un JSON valide au format exact suivant :
{{"score": 1}} pour pertinent
{{"score": 0}} pour non pertinent

Réponse JSON :"""

            self.prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["question", "documents"]
            )

            # LLM configuration for consistency
            self._grader_llm = ChatOllama(
                model=model_to_use,
                format="json",
                temperature=0.0,  # Deterministic for consistency
                num_predict=50,   # Short for efficiency
                timeout=10        # Short timeout
            )

            self._grader_chain = self.prompt | self._grader_llm | JsonOutputParser()

        except Exception as e:
            debugger.log_error(
                "Erreur d'initialisation du grader",
                "Grader initialization error",
                e
            )
            self._grader_llm = None
            self._grader_chain = None

    @debug_decorator(debugger, "grade_relevance",
                    "Évaluation de la pertinence d'un document",
                    "Document relevance evaluation")
    def grade_relevance(self, question: str, document: str,
                       model_name: Optional[str] = None,
                       use_cache: bool = True) -> Dict[str, Any]:
        """
        Evaluate document relevance for a question.
        
        Args:
            question: Question to evaluate
            document: Document content
            model_name: Optional LLM model to use
            use_cache: Whether to use caching (default: True)
            
        Returns:
            Dict with score (0 or 1) and metadata
        """
        start_time = time.time()
        
        # Cache verification
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(question, document)
            if cache_key in self._cache:
                cached_result, timestamp = self._cache[cache_key]
                if self._is_cache_valid(timestamp):
                    self.grading_stats["cache_hits"] += 1
                    self.grading_stats["total_grades"] += 1
                    
                    # Update average time (cache hit = minimal time)
                    cache_time = time.time() - start_time
                    self._update_timing_stats(cache_time)
                    
                    return {
                        **cached_result,
                        "cached": True,
                        "grading_time": cache_time
                    }
                else:
                    # Expired cache, remove
                    del self._cache[cache_key]

        # Initialize grader if necessary
        self._initialize_grader(model_name)
        
        if self._grader_chain is None:
            # Fallback: consider as relevant if grader not available
            self.grading_stats["errors"] += 1
            return {
                "score": 1,
                "error": "Grader not available",
                "grading_time": time.time() - start_time,
                "cached": False
            }

        try:
            # Truncate document if too long for optimization
            doc_truncated = document[:2000] if len(document) > 2000 else document
            
            # Evaluation by LLM
            result = self._grader_chain.invoke({
                "question": question,
                "documents": doc_truncated
            })
            
            # Validation and result cleaning
            score = int(result.get("score", 1))  # Default: relevant
            score = 1 if score > 0 else 0  # Binary normalization
            
            grading_time = time.time() - start_time
            
            # Update statistics
            self.grading_stats["total_grades"] += 1
            if score == 1:
                self.grading_stats["positive_grades"] += 1
            else:
                self.grading_stats["negative_grades"] += 1
                
            self._update_timing_stats(grading_time)
            
            # Final result
            final_result = {
                "score": score,
                "confidence": "high" if abs(score - 0.5) > 0.3 else "medium",
                "grading_time": grading_time,
                "document_length": len(document),
                "truncated": len(document) > 2000,
                "cached": False
            }
            
            # Caching
            if use_cache and cache_key:
                # Clean cache if too large
                if len(self._cache) >= self.cache_size:
                    self._cleanup_cache()
                    
                self._cache[cache_key] = (final_result.copy(), datetime.now().timestamp())
            
            return final_result

        except json.JSONDecodeError:
            # JSON parsing error - probably malformed response
            self.grading_stats["errors"] += 1
            debugger.log_warning(
                "Échec du parsing JSON du grader",
                "Grader JSON parsing failed"
            )
            return {
                "score": 1,  # Default: relevant on error
                "error": "JSON parsing failed",
                "grading_time": time.time() - start_time,
                "cached": False
            }

        except Exception as e:
            # Other errors
            self.grading_stats["errors"] += 1
            debugger.log_error(
                "Erreur lors de l'évaluation",
                "Error during grading evaluation",
                e
            )
            return {
                "score": 1,  # Default: relevant on error
                "error": str(e),
                "grading_time": time.time() - start_time,
                "cached": False
            }

    def _update_timing_stats(self, grading_time: float):
        """
        Update timing statistics.
        
        Args:
            grading_time: Time taken for grading
        """
        self.grading_stats["total_time"] += grading_time
        self.grading_stats["avg_time"] = (
            self.grading_stats["total_time"] /
            max(1, self.grading_stats["total_grades"])
        )

    def _cleanup_cache(self, keep_ratio: float = 0.7):
        """
        Clean up cache keeping most recent entries.
        
        Args:
            keep_ratio: Ratio of entries to keep (0.0-1.0)
        """
        if len(self._cache) <= self.cache_size * keep_ratio:
            return

        # Sort by timestamp (most recent first)
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1][1],  # timestamp
            reverse=True
        )

        # Keep only most recent
        keep_count = int(self.cache_size * keep_ratio)
        self._cache = dict(sorted_items[:keep_count])
        
        debugger.log_info(
            f"Cache nettoyé: {len(sorted_items) - keep_count} entrées supprimées",
            f"Cache cleaned: {len(sorted_items) - keep_count} entries removed"
        )

    def get_grading_stats(self) -> Dict[str, Any]:
        """
        Return detailed grader statistics.
        
        Returns:
            Dict with comprehensive grading statistics
        """
        stats = self.grading_stats.copy()
        
        if stats["total_grades"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_grades"]
            stats["positive_rate"] = stats["positive_grades"] / stats["total_grades"]
            stats["error_rate"] = stats["errors"] / stats["total_grades"]
        else:
            stats.update({
                "cache_hit_rate": 0.0,
                "positive_rate": 0.0,
                "error_rate": 0.0
            })

        stats.update({
            "cache_size": len(self._cache),
            "cache_max_size": self.cache_size,
            "cache_ttl_minutes": self.cache_ttl_minutes,
            "grader_initialized": self._grader_chain is not None
        })

        return stats

    def reset_stats(self):
        """Reset all statistics."""
        self.grading_stats = {
            "total_grades": 0,
            "cache_hits": 0,
            "positive_grades": 0,
            "negative_grades": 0,
            "errors": 0,
            "avg_time": 0.0,
            "total_time": 0.0
        }
        debugger.log_info("Statistiques du grader remises à zéro", "Grader stats reset")

    def clear_cache(self):
        """Clear cache completely."""
        self._cache.clear()
        debugger.log_info("Cache du grader vidé", "Grader cache cleared")


# Global optimized grader instance
_global_grader = RetrievalGrader()


def retrieval_grader(question: str, documents: str,
                    local_llm: Optional[str] = None) -> Dict[str, Any]:
    """
    Backward-compatible interface with the previous version.
    
    Args:
        question: Question to evaluate
        documents: Document content
        local_llm: Optional LLM model to use
        
    Returns:
        Dict with score (0 or 1) and metadata
    """
    return _global_grader.grade_relevance(
        question=question,
        document=documents,
        model_name=local_llm
    )


def get_grader_stats() -> Dict[str, Any]:
    """Return global grader statistics."""
    return _global_grader.get_grading_stats()


def reset_grader_stats():
    """Reset global grader statistics."""
    _global_grader.reset_stats()


def clear_grader_cache():
    """Clear global grader cache."""
    _global_grader.clear_cache()