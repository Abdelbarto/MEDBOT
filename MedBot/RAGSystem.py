# -*- coding: utf-8 -*-
"""
High-performance Medical RAG System with advanced validation and monitoring.

This module provides the main RAG system implementation with comprehensive
document processing, query handling, validation, and performance monitoring.

Features:
- Multi-profile configuration (performance/accuracy/balanced)
- Advanced document retrieval with intelligent filtering
- Response validation and quality assessment
- Performance monitoring and optimization
- Robust error handling and recovery
- Comprehensive logging and debugging

Author: Souleiman & Abdelbar Medical RAG System
Created: 2025
"""

import time
import json
from typing import Dict, Any, Optional, List

from memory_builder import MemoryBuilder
from llm_client import OllamaClient
from model_settings import Model_Settings
from debugging import get_debugger, debug_decorator
import re
# Initialize debugger
debugger = get_debugger()

# Conditional imports for advanced modules
try:
    from _engine import ResponseValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    debugger.log_warning("Engine de validation non disponible", "Validation engine not available")


try:
    from quality_monitor import QualityMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    debugger.log_warning("Monitoring de qualité non disponible", "Quality monitoring not available")


class RAGSystem:
    """
    High-performance Medical RAG System with comprehensive validation and monitoring.
    
    Integrates document processing, vector search, LLM generation, validation,
    and performance monitoring into a cohesive medical question-answering system.
    """

    def __init__(self, profile: str = "balanced", enable_validation: bool = None,
                 enable_monitoring: bool = None):
        """
        Initialize RAG system with specified configuration profile.
        
        Args:
            profile: Configuration profile ('performance', 'accuracy', 'balanced')
            enable_validation: Force validation engine on/off (overrides profile)
            enable_monitoring: Force monitoring on/off (overrides profile)
        """
        # Basic configuration
        self.settings = Model_Settings(profile)
        self.memory = MemoryBuilder(
            cache_enabled=getattr(self.settings, 'CACHE_ENABLED', True),
            settings=self.settings
        )
        self.memory.instantiate()
        self.llm = OllamaClient(model_name=self.settings.MODEL_NAME)

        # Advanced modules (conditional activation)
        self.validator = None
        self.monitor = None

        # Validation engine
        validation_enabled = (enable_validation if enable_validation is not None
                            else getattr(self.settings, 'VALIDATION_ENABLED', False))
        if validation_enabled and VALIDATION_AVAILABLE:
            self.validator = ResponseValidator(self.llm, self.settings)
            debugger.log_info("Engine de validation activé", "Validation engine activated")

        # Quality monitoring
        monitoring_enabled = (enable_monitoring if enable_monitoring is not None
                            else getattr(self.settings, 'MONITORING_ENABLED', True))
        if monitoring_enabled and MONITORING_AVAILABLE:
            self.monitor = QualityMonitor()
            debugger.log_info("Monitoring de qualité activé", "Quality monitoring activated")

        # Session statistics
        self.session_stats = {
            "start_time": time.time(),
            "total_queries": 0,
            "cache_hits": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }

        debugger.log_info(
            f"Système RAG initialisé avec le profil: {profile}",
            f"RAG System initialized with profile: {profile}"
        )

        if hasattr(self.settings, 'CACHE_ENABLED') and self.settings.CACHE_ENABLED:
            cache_stats = self.memory.get_cache_stats()
            debugger.log_info(
                f"Cache: {cache_stats.get('total_entries', 0)} entrées",
                f"Cache: {cache_stats.get('total_entries', 0)} entries"
            )
    @debug_decorator(debugger, "load_documents",
                    "Chargement des documents depuis un dossier",
                    "Load documents from folder")
    def load_documents(self, folder_path: str = 'uploaded_docs') -> bool:
        """
        Load PDF files from directory into vector store.
        
        Args:
            folder_path: Path to directory containing PDFs
            
        Returns:
            True if documents were loaded successfully
        """
        paths = self.memory.list_pdf_files(folder_path)
        
        if not paths:
            debugger.log_warning(f"Aucun fichier PDF trouvé dans {folder_path}", f"No PDF files found in {folder_path}")
            return False

        summary = self.memory.vectorstore_add_multi_files(paths)
        debugger.log_info(f"Documents chargés depuis {folder_path}", f"Documents loaded from {folder_path}")
        
        return True

    @debug_decorator(debugger, "load_all_documents",
                    "Chargement de tous les documents",
                    "Load all documents")
    def load_all_documents(self):
        """
        Load documents from both permanent and temporary folders.
        
        Returns:
            Tuple indicating success for (permanent, temporary) folders
        """
        permanent_loaded = self.load_documents('uploaded_docs/permanent')
        temporary_loaded = self.load_documents('uploaded_docs/temporary')

        if permanent_loaded or temporary_loaded:
            perm_sources = self.memory.list_sources_by_type("permanent")
            temp_sources = self.memory.list_sources_by_type("temporary")
            debugger.log_info(
                f"Chargement terminé: {len(perm_sources)} permanents, {len(temp_sources)} temporaires",
                f"Loading complete: {len(perm_sources)} permanent, {len(temp_sources)} temporary"
            )
        else:
            debugger.log_warning("Aucun document n'a été chargé", "No documents were loaded")

        return permanent_loaded, temporary_loaded

    @debug_decorator(debugger, "ask_question",
                    "Traitement d'une question avec le pipeline RAG complet",
                    "Process question with complete RAG pipeline")
    def ask_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        retrieval_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        bypass_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Process question with complete optimized RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of passages to retrieve (default from settings)
            retrieval_threshold: Similarity threshold (default from settings)
            where: Search filters (e.g., {"source": "file.pdf"})
            debug: Enable debug logging
            bypass_cache: Force bypass of caching
            
        Returns:
            Dict containing answer, passages, validation, and metrics
        """
        start_time = time.time()

        # Default parameters from settings
        top_k = top_k or self.settings.RETRIEVAL_TOP_K
        retrieval_threshold = retrieval_threshold or self.settings.RETRIEVAL_THRESHOLD

        # Temporary cache bypass if requested
        original_cache_setting = None
        if bypass_cache and hasattr(self.memory, 'cache_enabled'):
            original_cache_setting = self.memory.cache_enabled
            self.memory.cache_enabled = False

        try:
            # Phase 1: Document retrieval
            retrieval_start = time.time()
            
            if debug:
                debugger.log_debug(f"Début de récupération: top_k={top_k}, threshold={retrieval_threshold}",
                                  f"Starting retrieval: top_k={top_k}, threshold={retrieval_threshold}")
                if where:
                    debugger.log_debug(f"Filtres appliqués: {where}", f"Filters applied: {where}")

            # CRITICAL CORRECTION: Retrieval with robust validation
            try:
                results = self.memory.vectorstore_similarity_search_with_score(
                    question=question,
                    k=top_k,
                    retrieval_threshold=retrieval_threshold,
                    where=where,
                    debug=debug
                )

                if debug:
                    debugger.log_debug(f"Récupéré {len(results)} résultats", f"Retrieved {len(results)} results")

            except Exception as e:
                debugger.log_error("ERREUR CRITIQUE lors de la recherche de similarité", 
                                 "CRITICAL ERROR during similarity search", e)
                
                # Return structured error
                return {
                    "question": question,
                    "answer": f"Erreur critique lors de la recherche de similarité: {str(e)}",
                    "relevant_passages": [],
                    "validation": {"error": f"Search error: {str(e)}"},
                    "performance": {
                        "total_time": time.time() - start_time,
                        "error": True
                    },
                    "metadata": {
                        "timestamp": time.time(),
                        "error": True,
                        "error_type": "similarity_search_error"
                    }
                }

            retrieval_time = time.time() - retrieval_start

            # CRITICAL VALIDATION: Ensure results is a valid list of tuples
            if not isinstance(results, list):
                if debug:
                    debugger.log_error(f"CRITIQUE: Results n'est pas une liste: {type(results)}", 
                                     f"CRITICAL: Results is not a list: {type(results)}")
                results = []

            # Validate each result
            validated_results = []
            for i, result in enumerate(results):
                try:
                    if not isinstance(result, (tuple, list)) or len(result) != 2:
                        if debug:
                            debugger.log_warning(f"Format de résultat invalide à l'index {i}: {type(result)}",
                                               f"Invalid result format at index {i}: {type(result)}")
                        continue

                    doc, score = result

                    if not hasattr(doc, 'metadata'):
                        if debug:
                            debugger.log_warning(f"Document à l'index {i} sans metadata: {type(doc)}",
                                               f"Document at index {i} has no metadata: {type(doc)}")
                        continue

                    if not isinstance(score, (int, float)):
                        if debug:
                            debugger.log_warning(f"Score à l'index {i} non numérique: {type(score)}",
                                               f"Score at index {i} is not numeric: {type(score)}")
                        continue

                    validated_results.append((doc, score))

                except Exception as e:
                    if debug:
                        debugger.log_warning(f"Erreur de validation du résultat {i}", f"Error validating result {i}")
                    continue

            if debug and len(validated_results) != len(results):
                debugger.log_warning(f"CRITIQUE: {len(results) - len(validated_results)} résultats invalides filtrés",
                                    f"CRITICAL: {len(results) - len(validated_results)} invalid results filtered out")

            # Phase 2: Response generation
            generation_start = time.time()

            # CRITICAL CORRECTION: Use validated results
            try:
                relevant_passages = []
                for i, (doc, score) in enumerate(validated_results, start=1):
                    relevant_passages.append((i, (doc, score)))
                
                # CORRECTION: Utiliser 'relevant_passages' au lieu de 'relevant'
                if debug:
                    debugger.log_debug(f"Préparation de la génération avec {len(relevant_passages)} passages pertinents",
                                    f"Preparing to generate response with {len(relevant_passages)} relevant passages")
                    for i, (doc, score) in relevant_passages[:3]: # Log first 3
                        source = doc.metadata.get("source", "Unknown") if hasattr(doc, 'metadata') else "No metadata"
                        debugger.log_debug(f"{i}: {source} (score: {score:.3f})", f"{i}: {source} (score: {score:.3f})")
                
                # Assigner la variable pour compatibilité avec le reste du code
                relevant = relevant_passages
                
            except Exception as e:
                debugger.log_error("ERREUR CRITIQUE dans enumerate", "CRITICAL ERROR in enumerate", e)
                relevant = []


            if relevant:
                try:
                    # Build context from documents
                    context_parts = []
                    for i, (doc, _) in relevant:
                        if hasattr(doc, 'page_content'):
                            context_parts.append(f"[{i}] {doc.page_content}")
                        else:
                            context_parts.append(f"[{i}] [Document without content]")

                    context = "\n\n".join(context_parts)
                    prompt = self._create_enhanced_rag_prompt(question, context, relevant)

                    if debug:
                        debugger.log_debug(f"Génération de réponse avec contexte de {len(context)} caractères",
                                          f"Generating response with context length: {len(context)} chars")

                    response = self.llm.chat([{"role": "user", "content": prompt}])
                    answer = response["choices"][0]["message"]["content"].strip()

                except Exception as e:
                    debugger.log_error("ERREUR lors de la génération de réponse", "ERROR in response generation", e)
                    answer = f"Erreur lors de la génération de la réponse: {str(e)}"
            else:
                if debug:
                    debugger.log_warning("Aucun passage pertinent trouvé, utilisation du fallback intelligent",
                                        "No relevant passages found, using intelligent fallback")
                
                # NOUVEAU : Fallback intelligent au lieu de message générique
                fallback_prompt = f"""Tu es un assistant médical expert. 

            QUESTION : {question}

            SITUATION : Aucun document spécifique trouvé pour cette question.

            INSTRUCTIONS :
            - Utilise tes connaissances médicales générales pour donner une réponse utile
            - Indique clairement que tu n'as pas de documents spécifiques sur ce sujet
            - Donne des conseils généraux ou oriente vers des ressources appropriées
            - Reste dans ton domaine de compétence médicale
            - Maximum 200 mots

            RÉPONSE GÉNÉRALE :"""
                
                try:
                    response = self.llm.chat([{"role": "user", "content": fallback_prompt}])
                    answer = response["choices"][0]["message"]["content"].strip()
                    answer = "⚠️ Réponse basée sur les connaissances générales (pas de documents spécifiques) :\n\n" + answer
                except Exception as e:
                    answer = "Information insuffisante dans les documents disponibles pour répondre à cette question spécifique."

            generation_time = time.time() - generation_start

            # Phase 3: Validation (if enabled)
            validation_results = {}
            if self.validator:
                try:
                    if debug:
                        debugger.log_debug("Exécution de la validation de réponse", "Running response validation")
                    
                    validation_results = self.validator.validate_medical_response(
                        answer, validated_results, question
                    )

                except Exception as e:
                    if debug:
                        debugger.log_warning("Erreur de validation", "Validation error")
                    validation_results = {"error": str(e), "confidence_level": "unknown"}

            # Performance metrics calculation
            total_time = time.time() - start_time

            # Update session statistics
            self.session_stats["total_queries"] += 1
            self.session_stats["total_response_time"] += total_time
            self.session_stats["avg_response_time"] = (
                self.session_stats["total_response_time"] /
                self.session_stats["total_queries"]
            )

            # Quality logging (if enabled)
            if self.monitor:
                try:
                    self.monitor.log_query_performance(
                        question=question,
                        retrieval_time=retrieval_time,
                        generation_time=generation_time,
                        validation_results=validation_results or {}
                    )

                except Exception as e:
                    if debug:
                        debugger.log_warning("Erreur de monitoring", "Monitoring error")

            # Build final response
            result = {
                "question": question,
                "answer": answer,
                "relevant_passages": relevant,
                "validation": validation_results,
                "performance": {
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                    "passages_found": len(relevant),
                    "used_cache": False  # Will be updated by cache system
                },
                "metadata": {
                    "settings_profile": self.settings.profile,
                    "timestamp": time.time(),
                    "filters_applied": where is not None
                }
            }

            # Add improvement suggestions if validation available
            if self.validator and validation_results:
                suggestions = self.validator.generate_improvement_suggestions(validation_results)
                if suggestions:
                    result["improvement_suggestions"] = suggestions

            if debug:
                debugger.log_info(f"Requête terminée en {total_time:.3f}s", f"Query completed in {total_time:.3f}s")
                if validation_results:
                    debugger.log_info(f"Confiance: {validation_results.get('confidence_level', 'unknown')}",
                                     f"Confidence: {validation_results.get('confidence_level', 'unknown')}")

            return result

        except Exception as e:
            error_time = time.time() - start_time
            debugger.log_error("ERREUR INATTENDUE dans ask_question", "UNEXPECTED ERROR in ask_question", e)

            if self.monitor:
                try:
                    self.monitor.log_query_performance(
                        question=question,
                        retrieval_time=0,
                        generation_time=0,
                        validation_results={},
                        error_message=str(e)
                    )
                except:
                    pass

            return {
                "question": question,
                "answer": f"Erreur inattendue lors du traitement de la question: {str(e)}",
                "relevant_passages": [],
                "validation": {"error": str(e)},
                "performance": {
                    "total_time": round(error_time, 3),
                    "error": True
                },
                "metadata": {
                    "timestamp": time.time(),
                    "error": True,
                    "error_type": "unexpected_error"
                }
            }

        finally:
            # Restore cache settings
            if bypass_cache and original_cache_setting is not None:
                self.memory.cache_enabled = original_cache_setting

    def _create_enhanced_rag_prompt(self, question: str, context: str,
                                passages_metadata: List) -> str:
        """
        Create adaptive prompt that lets LLM find its own path.
        
        Args:
            question: User's question
            context: Retrieved document context  
            passages_metadata: Metadata for passages
        
        Returns:
            Adaptive prompt string that gives LLM maximum flexibility
        """
        import re
        
        # Build citable references
        citations = []
        for i, (doc, score) in passages_metadata:
            if hasattr(doc, 'metadata'):
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                doc_type = doc.metadata.get("doc_type", "unknown")
                citations.append(f"[{i}] {src} (p.{page})")
            else:
                citations.append(f"[{i}] Document sans métadonnées")
        
        # Detect if it's a QCM (keep this detection for QCM handling)
        is_qcm = bool(re.search(r'[abcd]\s*[.)]\s*\w+', question.lower()))
        
        if is_qcm:
            # QCM-specific prompt (keep this part)
            return f"""Tu es un expert médical qui répond aux QCM avec précision absolue.

    CONTEXTE DOCUMENTÉ :
    {context}

    QUESTION QCM : {question}

    SOURCES DISPONIBLES :
    {chr(10).join(citations)}

    INSTRUCTIONS QCM :
    - Analyse chaque option A, B, C, D proposée
    - Identifie la définition médicale correcte basée sur les documents
    - Réponds au format: "La bonne réponse est [LETTRE]. [Justification courte avec citations]"
    - Maximum 150 mots
    - Sois précis et factuel

    RÉPONSE :"""
        
        else:
            # NOUVEAU : Prompt adaptatif universel
            return f"""Tu es un assistant médical expert avec une intelligence adaptive. 

    CONTEXTE DOCUMENTÉ :
    {context}

    SOURCES DISPONIBLES POUR CITATION :
    {chr(10).join(citations)}

    QUESTION : {question}

    INSTRUCTIONS ADAPTATIVES :
    🧠 RAISONNEMENT AUTONOME :
    - Analyse la question et détermine la meilleure approche de réponse
    - Adapte ton style (technique, vulgarisé, comparatif, etc.) selon le besoin
    - Structure ta réponse comme tu le juges approprié
    - Si la question est inhabituelle, créative ou complexe : sois innovant dans ton approche

    📚 UTILISATION DES SOURCES :
    - Base-toi sur les documents fournis
    - Cite tes sources avec [numéro] quand pertinent  
    - Si info manquante : "Information non disponible dans les documents pour [aspect]"

    🎯 QUALITÉ DE RÉPONSE :
    - Privilégie la précision et l'utilité
    - Adapte la longueur selon la complexité (50-300 mots)
    - Sois direct et actionnable
    - Gère les questions bizarres/inattendues avec intelligence

    ⚡ LIBERTÉ TOTALE :
    - Pas de format imposé - choisis la structure optimale
    - Questions multiples → réponses structurées
    - Questions techniques → réponses détaillées  
    - Questions simples → réponses concises
    - Questions créatives → réponses créatives

    RÉPONSE ADAPTIVE :"""


    def display_result(self, result: Dict[str, Any], show_performance: bool = True,
                      show_validation: bool = True):
        """
        Display results with performance and validation details.
        
        Args:
            result: Result dictionary from ask_question
            show_performance: Whether to show performance metrics
            show_validation: Whether to show validation results
        """
        # Main response display
        print(f"\n💬 Réponse: {result['answer']}\n")

        # Sources display
        if not result["relevant_passages"]:
            print("⚠️ Aucune source trouvée.")
        else:
            print("📄 Sources des passages pertinents:")
            cpt = 0
            for idx, (doc, score) in result["relevant_passages"]:
                if cpt < 3:
                    if hasattr(doc, 'metadata'):
                        src = doc.metadata.get("source", "Source inconnue")
                        page = doc.metadata.get("page", "N/A")
                        doc_type = doc.metadata.get("doc_type", "unknown")

                        # Parse headings
                        headings = doc.metadata.get("headings", [])
                        if isinstance(headings, str):
                            try:
                                headings = json.loads(headings)
                            except:
                                headings = []

                        section = headings[-1] if headings else "Section inconnue"
                        type_icon = "🔒" if doc_type == "permanent" else "⏰" if doc_type == "temporary" else "❓"

                        print(f" • {type_icon} {src} | p.{page} | {section} (score {score:.3f})")
                    cpt += 1

        # Performance metrics display
        if show_performance and "performance" in result:
            perf = result["performance"]
            print(f"\n⚡ Performance:")
            print(f" • Temps total: {perf.get('total_time', 0):.3f}s")
            print(f" • Récupération: {perf.get('retrieval_time', 0):.3f}s")
            print(f" • Génération: {perf.get('generation_time', 0):.3f}s")
            print(f" • Passages trouvés: {perf.get('passages_found', 0)}")

            if perf.get('used_cache'):
                print(" • 💾 Résultat mis en cache")

        # Validation display
        if show_validation and "validation" in result and result["validation"]:
            validation = result["validation"]
            if "confidence_level" in validation:
                confidence = validation["confidence_level"]
                confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
                print(f"\n{confidence_emoji} Niveau de confiance: {confidence.upper()}")

                # Validation details if available
                if "citation_accuracy" in validation:
                    cit_acc = validation["citation_accuracy"]
                    if cit_acc.get("found_citations", 0) > 0:
                        print(f" 📚 Citations: {cit_acc['found_citations']}")

                if "medical_consistency" in validation:
                    med_cons = validation["medical_consistency"]
                    if med_cons.get("valid"):
                        print(" ✅ Cohérence médicale validée")
                    elif not med_cons.get("validation_successful", True):
                        print(" ⚠️ Validation médicale non disponible")

        # Improvement suggestions
        if "improvement_suggestions" in result and result["improvement_suggestions"]:
            print(f"\n💡 Suggestions d'amélioration:")
            for suggestion in result["improvement_suggestions"]:
                print(f" • {suggestion}")

    @debug_decorator(debugger, "get_system_status",
                    "Récupération du statut du système",
                    "Get system status")
    def get_system_status(self) -> Dict[str, Any]:
        """
        Return comprehensive system status report.
        
        Returns:
            Dict with detailed system status information
        """
        status = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.session_stats["start_time"],
            "settings": {
                "profile": self.settings.profile,
                "model": self.settings.MODEL_NAME,
                "cache_enabled": getattr(self.settings, 'CACHE_ENABLED', False),
                "validation_enabled": self.validator is not None,
                "monitoring_enabled": self.monitor is not None
            },
            "session_stats": self.session_stats.copy(),
            "memory": {
                "total_sources": len(self.memory.list_sources_in_db()),
                "permanent_sources": len(self.memory.list_sources_by_type("permanent")),
                "temporary_sources": len(self.memory.list_sources_by_type("temporary"))
            }
        }

        # Add cache statistics
        if hasattr(self.memory, 'get_cache_stats'):
            status["cache"] = self.memory.get_cache_stats()

        # Add quality report
        if self.monitor:
            try:
                quality_report = self.monitor.get_quality_report(days_back=1)
                status["quality"] = {
                    "overall_score": quality_report.get("quality_score", {}),
                    "system_health": quality_report.get("system_health", "UNKNOWN"),
                    "recommendations": quality_report.get("recommendations", [])
                }
            except:
                status["quality"] = {"error": "Quality report unavailable"}

        return status

    @debug_decorator(debugger, "optimize_performance",
                    "Optimisation automatique des performances",
                    "Automatic performance optimization")
    def optimize_performance(self):
        """
        Automatically optimize system performance.
        
        Returns:
            Dict with optimization results and new system status
        """
        optimizations_performed = []

        # Cache cleanup
        if hasattr(self.memory, 'clean_expired_cache'):
            self.memory.clean_expired_cache()
            optimizations_performed.append("Cache cleanup")

        # System status before optimization
        status = self.get_system_status()

        # Automatic threshold adjustment if many slow queries
        if (self.session_stats["total_queries"] > 10 and
            self.session_stats["avg_response_time"] > 8.0):
            
            # Reduce number of retrieved passages
            if self.settings.RETRIEVAL_TOP_K > 10:
                self.settings.RETRIEVAL_TOP_K = max(10, self.settings.RETRIEVAL_TOP_K - 3)
                optimizations_performed.append("Reduced RETRIEVAL_TOP_K")

            # Increase similarity threshold
            if self.settings.RETRIEVAL_THRESHOLD < 0.15:
                self.settings.RETRIEVAL_THRESHOLD = min(0.15, self.settings.RETRIEVAL_THRESHOLD + 0.02)
                optimizations_performed.append("Increased RETRIEVAL_THRESHOLD")

        # Optimization report
        debugger.log_info(
            f"Optimisations effectuées: {', '.join(optimizations_performed) if optimizations_performed else 'Aucune nécessaire'}",
            f"Optimizations completed: {', '.join(optimizations_performed) if optimizations_performed else 'None needed'}"
        )

        return {
            "optimizations_performed": optimizations_performed,
            "system_status_after": self.get_system_status()
        }


def main():
    """Entry point for manual RAG system testing."""
    pass


if __name__ == "__main__":
    main()