# -*- coding: utf-8 -*-
"""
Optimized LLM client with error handling and performance metrics.

This module provides abstract base classes and implementations for LLM clients
with comprehensive error handling, retry mechanisms, and performance tracking.

Features:
- Abstract base class for LLM clients
- Ollama client with retry logic
- Performance metrics and statistics
- Robust error handling and logging
- Client management utilities

Author: Souleiman & Abdelbar Medical RAG System
Created: 2025
"""

from typing import Any, Dict, List, Optional, Union
import time
import json
import os
from datetime import datetime
from debugging import get_debugger, debug_decorator

# Initialize debugger
debugger = get_debugger()


class BaseLLMClient:
    """
    Abstract base class for LLM clients with integrated error handling
    and performance metrics.
    
    Provides common functionality for all LLM client implementations including
    statistics tracking, error handling, and logging capabilities.
    """

    def __init__(self, model_name: str):
        """
        Initialize the base LLM client.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.call_count = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.error_count = 0
        self.last_error = None

        # Create logs directory if it doesn't exist
        self.logs_dir = "system_files/logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        debugger.log_info(
            f"Client LLM de base initialisé pour {model_name}",
            f"Base LLM client initialized for {model_name}"
        )

    def chat(self, messages: List[Dict[str, str]],
             functions: Optional[List[Dict[str, Any]]] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Main interface for LLM interaction.
        
        Args:
            messages: List of conversation messages
            functions: Optional function definitions
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary from LLM
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the `chat` method.")

    def bind(self) -> "BaseLLMClient":
        """Return client instance for fluent chaining."""
        return self

    def get_stats(self) -> Dict[str, Any]:
        """
        Return client usage statistics.
        
        Returns:
            Dict containing comprehensive usage statistics
        """
        return {
            "model_name": self.model_name,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_time_per_call": self.total_time / max(1, self.call_count),
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.call_count),
            "last_error": self.last_error
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.call_count = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.error_count = 0
        self.last_error = None
        
        debugger.log_info(
            "Statistiques du client remises à zéro",
            "Client statistics reset"
        )

    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Generate standardized error response.
        
        Args:
            error_message: Error message to format
            
        Returns:
            Standardized error response dictionary
        """
        self.error_count += 1
        self.last_error = {
            "timestamp": datetime.now().isoformat(),
            "message": error_message
        }

        return {
            "choices": [{
                "message": {
                    "content": f"Erreur: {error_message}. Veuillez réessayer ou utiliser un modèle différent."
                }
            }],
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }

    def _log_call(self, start_time: float, success: bool = True, tokens_used: int = 0):
        """
        Log call metrics.
        
        Args:
            start_time: Call start timestamp
            success: Whether the call succeeded
            tokens_used: Number of tokens consumed
        """
        call_time = time.time() - start_time
        self.call_count += 1
        self.total_time += call_time

        if success:
            self.total_tokens += tokens_used


class OllamaClient(BaseLLMClient):
    """
    Optimized LLM client for Ollama with temperature handling,
    automatic retry, and detailed metrics.
    
    Provides robust interface to Ollama models with comprehensive error
    handling, retry logic, and performance monitoring.
    """

    def __init__(self, model_name: str, max_retries: int = 2, retry_delay: float = 1.0):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the Ollama model
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries in seconds
        """
        super().__init__(model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.init_error = None
        self._initialize_client()

    @debug_decorator(debugger, "initialize_ollama_client",
                    "Initialisation du client Ollama",
                    "Ollama client initialization")
    def _initialize_client(self):
        """Initialize ChatOllama client with error handling."""
        try:
            from langchain_ollama import ChatOllama
            from model_settings import Model_Settings as ModelSettings

            cfg = ModelSettings()
            self.client = ChatOllama(
                model=self.model_name,
                temperature=cfg.TEMPERATURE,
                top_k=cfg.TOP_K,
                top_p=cfg.TOP_P,
                num_predict=cfg.NUM_PREDICT,
                repeat_penalty=cfg.REPEAT_PENALTY,
                stream=False,  # Disabled for more stable responses
                timeout=30,    # Timeout to avoid blocking
            )

            # Store config for reference
            self.config = {
                "temperature": cfg.TEMPERATURE,
                "top_k": cfg.TOP_K,
                "top_p": cfg.TOP_P,
                "num_predict": cfg.NUM_PREDICT,
                "repeat_penalty": cfg.REPEAT_PENALTY
            }

        except Exception as e:
            error_msg = f"Error initializing Ollama client: {e}"
            debugger.log_error(
                "Erreur d'initialisation du client Ollama",
                "Ollama client initialization error",
                e
            )
            self.client = None
            self.init_error = str(e)

    @debug_decorator(debugger, "ollama_chat",
                    "Envoi d'une conversation au modèle Ollama",
                    "Send conversation to Ollama model")
    def chat(self, messages: List[Dict[str, str]],
             functions: Optional[List[Dict[str, Any]]] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Send conversation to Ollama model with automatic retry.
        
        Args:
            messages: Conversation messages
            functions: Optional function definitions (unused)
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with model output
        """
        start_time = time.time()
        
        if self.client is None:
            return self._format_error_response(
                f"Client Ollama non initialisé: {self.init_error or 'Erreur inconnue'}"
            )

        # Build prompt from messages
        prompt = self._build_prompt_from_messages(messages)

        # Retry attempts
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    debugger.log_warning(
                        f"Tentative de retry {attempt}/{self.max_retries}",
                        f"Retry attempt {attempt}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay * attempt)

                # Dynamic parameter adjustment if specified
                client_to_use = self.client
                if temperature is not None or max_tokens is not None:
                    client_to_use = self._get_adjusted_client(temperature, max_tokens)

                # Model call
                response = client_to_use.invoke(prompt)

                # Token estimation (approximative)
                tokens_used = len(prompt.split()) + len(response.content.split())
                self._log_call(start_time, success=True, tokens_used=tokens_used)

                return {
                    "choices": [{
                        "message": {
                            "content": response.content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(response.content.split()),
                        "total_tokens": tokens_used
                    },
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "response_time": time.time() - start_time,
                    "attempt": attempt + 1
                }

            except Exception as e:
                last_error = str(e)
                if attempt == self.max_retries:
                    # Last attempt failed
                    self._log_call(start_time, success=False)
                    debugger.log_error(
                        f"Échec après {self.max_retries + 1} tentatives",
                        f"Failed after {self.max_retries + 1} attempts",
                        e
                    )
                    return self._format_error_response(
                        f"Échec après {self.max_retries + 1} tentatives: {last_error}"
                    )

                # Continue to next retry
                continue

        # Should never happen
        self._log_call(start_time, success=False)
        return self._format_error_response(f"Erreur inattendue: {last_error}")

    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Build optimized prompt from conversation messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        if not messages:
            return ""

        # Handle system, user, assistant roles
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user").lower()
            content = message.get("content", "").strip()
            
            if not content:
                continue

            if role == "system":
                prompt_parts.append(f"{content}")
            elif role == "user":
                prompt_parts.append(f"{content}")
            elif role == "assistant":
                prompt_parts.append(f"{content}")
            else:
                # Unknown role, treat as user
                prompt_parts.append(f"{content}")

        # Add response tag if last message isn't assistant
        if messages[-1].get("role", "").lower() != "assistant":
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    def _get_adjusted_client(self, temperature: Optional[float] = None,
                           max_tokens: Optional[int] = None):
        """
        Create temporary client with adjusted parameters.
        
        Args:
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            ChatOllama client with adjusted parameters
        """
        try:
            from langchain_ollama import ChatOllama

            # Use current parameters as base
            config = self.config.copy()
            
            if temperature is not None:
                config["temperature"] = max(0.0, min(2.0, temperature))
                
            if max_tokens is not None:
                config["num_predict"] = max(1, min(8192, max_tokens))

            return ChatOllama(
                model=self.model_name,
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                num_predict=config["num_predict"],
                repeat_penalty=config["repeat_penalty"],
                stream=False,
                timeout=30
            )

        except:
            # Fallback to main client
            return self.client

    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Ollama model.
        
        Returns:
            Dict with connection test results
        """
        test_start = time.time()
        try:
            result = self.chat([{
                "role": "user",
                "content": "Réponds simplement par 'OK' pour tester la connexion."
            }])
            
            success = "OK" in result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            test_result = {
                "success": success,
                "response_time": time.time() - test_start,
                "model": self.model_name,
                "error": result.get("error"),
                "config": self.config
            }
            
            debugger.log_info(
                f"Test de connexion: {'succès' if success else 'échec'}",
                f"Connection test: {'success' if success else 'failure'}"
            )
            
            return test_result

        except Exception as e:
            debugger.log_error(
                "Erreur lors du test de connexion",
                "Connection test error",
                e
            )
            return {
                "success": False,
                "response_time": time.time() - test_start,
                "model": self.model_name,
                "error": str(e),
                "config": self.config
            }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Return detailed statistics including configuration.
        
        Returns:
            Dict with comprehensive statistics
        """
        base_stats = self.get_stats()
        base_stats.update({
            "config": self.config,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "client_initialized": self.client is not None,
            "init_error": self.init_error
        })
        return base_stats


# Factory pattern for creating clients
def create_llm_client(client_type: str = "ollama", model_name: str = "llama3.2:1b",
                     **kwargs) -> BaseLLMClient:
    """
    Factory for creating LLM clients by type.
    
    Args:
        client_type: Type of client to create
        model_name: Model name to use
        **kwargs: Additional client parameters
        
    Returns:
        Configured LLM client instance
        
    Raises:
        ValueError: For unsupported client types
    """
    if client_type.lower() == "ollama":
        return OllamaClient(model_name, **kwargs)
    else:
        raise ValueError(f"Client type non supporté: {client_type}")


# Utilities for client management
class LLMClientManager:
    """Centralized manager for multiple LLM clients."""

    def __init__(self):
        """Initialize client manager."""
        self.clients = {}
        self.default_client = None
        
        debugger.log_info(
            "Gestionnaire de clients LLM initialisé",
            "LLM client manager initialized"
        )

    def register_client(self, name: str, client: BaseLLMClient, set_as_default: bool = False):
        """
        Register an LLM client.
        
        Args:
            name: Client identifier
            client: LLM client instance
            set_as_default: Whether to set as default client
        """
        self.clients[name] = client
        
        if set_as_default or self.default_client is None:
            self.default_client = client
            
        debugger.log_info(
            f"Client '{name}' enregistré",
            f"Client '{name}' registered"
        )

    def get_client(self, name: str = None) -> Optional[BaseLLMClient]:
        """
        Retrieve client by name or default client.
        
        Args:
            name: Optional client name
            
        Returns:
            LLM client instance or None
        """
        if name is None:
            return self.default_client
        return self.clients.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Return statistics for all clients.
        
        Returns:
            Dict mapping client names to their statistics
        """
        return {
            name: client.get_stats()
            for name, client in self.clients.items()
        }

    def test_all_connections(self) -> Dict[str, Any]:
        """
        Test connection for all Ollama clients.
        
        Returns:
            Dict mapping client names to test results
        """
        results = {}
        for name, client in self.clients.items():
            if isinstance(client, OllamaClient):
                results[name] = client.test_connection()
        return results