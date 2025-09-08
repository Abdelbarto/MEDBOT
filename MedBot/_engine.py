# -*- coding: utf-8 -*-
"""
Post-generation validation engine for medical accuracy and reliability.

This module provides comprehensive validation of medical responses generated
by the RAG system, ensuring medical accuracy, citation correctness, and
factual alignment with source documents.

Features:
- Medical consistency validation via LLM
- Citation accuracy checking
- Factual alignment verification
- Response completeness assessment
- Improvement suggestions generation

Author: Souleiman & Abdelbar Medical RAG System
Created: 2025
"""

from typing import Dict, List, Tuple, Any, Optional
import re
import json
import hashlib
import os
from datetime import datetime
from grader import retrieval_grader
from debugging import get_debugger, debug_decorator

# Initialize debugger
debugger = get_debugger()


class ResponseValidator:
    """
    Post-generation validation engine for ensuring medical rigor and accuracy.
    
    Validates medical responses against multiple criteria including citation
    accuracy, medical consistency, factual alignment, and completeness.
    """

    def __init__(self, llm_client, settings):
        """Initialize the response validator with LLM client and settings."""
        self.llm_client = llm_client
        self.settings = settings
        self.validation_cache = {}
        
        # Create cache directory if it doesn't exist
        self.cache_dir = "system_files/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        debugger.log_info(
            "Moteur de validation initialisé", 
            "Response validation engine initialized"
        )

    @debug_decorator(debugger, "validate_medical_response", 
                    "Validation complète d'une réponse médicale",
                    "Complete medical response validation")
    def validate_medical_response(self, response: str, passages: List, question: str) -> Dict[str, Any]:
        """
        Complete validation of a medical response.
        
        Args:
            response: Generated response to validate
            passages: Source passages used for generation
            question: Original question asked
            
        Returns:
            Dict containing validation results and confidence level
        """
        # Cache key to avoid repeated validations
        cache_key = hashlib.md5(f"{response}_{question}".encode()).hexdigest()
        
        if cache_key in self.validation_cache:
            debugger.log_info("Utilisation du cache de validation", "Using validation cache")
            return self.validation_cache[cache_key]

        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "citation_accuracy": self._check_citations(response, passages),
            "medical_consistency": self._check_medical_consistency(response, question),
            "factual_alignment": self._check_factual_alignment(response, passages),
            "completeness_score": self._check_completeness(response, question),
            "confidence_level": "low",
            "issues": [],
            "recommendations": []
        }

        # Calculate global confidence level
        validation_results["confidence_level"] = self._calculate_confidence_level(validation_results)
        
        # Cache the result
        self.validation_cache[cache_key] = validation_results
        
        debugger.log_info(
            f"Validation terminée - Confiance: {validation_results['confidence_level']}",
            f"Validation completed - Confidence: {validation_results['confidence_level']}"
        )
        
        return validation_results

    def _check_citations(self, response: str, passages: List) -> Dict[str, Any]:
        """
        Verify citation accuracy and completeness.
        
        Args:
            response: Response text to check
            passages: Available source passages
            
        Returns:
            Dict with citation validation results
        """
        citations = re.findall(r'\[(\d+)\]', response)
        citation_numbers = [int(c) for c in citations if c.isdigit()]
        valid_citations = all(num < len(passages) for num in citation_numbers)

        # Check if each important statement has a citation
        sentences = [s.strip() for s in re.split(r'[.!?]', response) if s.strip()]
        uncited_sentences = []
        
        for sentence in sentences:
            if len(sentence) > 20 and not re.search(r'\[\d+\]', sentence):
                # Exclude generic phrases
                if not any(generic in sentence.lower() for generic in [
                    "bonjour", "merci", "au revoir", "information insuffisante",
                    "veuillez consulter", "selon les documents"
                ]):
                    uncited_sentences.append(sentence[:50] + "...")

        citation_coverage = max(0, (len(sentences) - len(uncited_sentences)) / len(sentences)) if sentences else 0
        
        result = {
            "valid": valid_citations and len(uncited_sentences) < 2,
            "found_citations": len(citation_numbers),
            "invalid_citations": [num for num in citation_numbers if num >= len(passages)],
            "uncited_sentences": uncited_sentences,
            "citation_coverage": citation_coverage
        }
        
        debugger.log_debug(
            f"Vérification des citations: {len(citation_numbers)} trouvées",
            f"Citation check: {len(citation_numbers)} found",
            result
        )
        
        return result

    def _check_medical_consistency(self, response: str, question: str) -> Dict[str, Any]:
        """
        Verify medical consistency via LLM validation.
        
        Args:
            response: Response to validate
            question: Original question
            
        Returns:
            Dict with medical consistency results
        """
        consistency_prompt = f"""
Tu es un médecin expert chargé de valider la cohérence médicale d'une réponse.

QUESTION ORIGINALE: {question}

RÉPONSE À ÉVALUER: {response}

Évalue selon ces critères:
1. Cohérence terminologique médicale
2. Logique diagnostique ou thérapeutique
3. Absence de contradictions
4. Respect des bonnes pratiques

Réponds UNIQUEMENT par ce JSON:
{{
    "consistent": true/false,
    "medical_accuracy": "high"/"medium"/"low",
    "issues": ["problème1", "problème2"],
    "terminology_correct": true/false
}}
"""

        try:
            result = self.llm_client.chat([{"role": "user", "content": consistency_prompt}])
            content = result["choices"][0]["message"]["content"].strip()
            
            # Robust JSON parsing
            try:
                parsed = json.loads(content)
                validation_result = {
                    "valid": parsed.get("consistent", False),
                    "medical_accuracy": parsed.get("medical_accuracy", "low"),
                    "issues": parsed.get("issues", []),
                    "terminology_correct": parsed.get("terminology_correct", False),
                    "validation_successful": True
                }
            except json.JSONDecodeError:
                # Fallback: simple text analysis
                validation_result = {
                    "valid": "consistent" in content.lower() or "cohérent" in content.lower(),
                    "medical_accuracy": "unknown",
                    "issues": ["Parsing validation failed"],
                    "terminology_correct": False,
                    "validation_successful": False
                }
                debugger.log_warning(
                    "Échec du parsing JSON pour la validation médicale",
                    "JSON parsing failed for medical validation"
                )

        except Exception as e:
            validation_result = {
                "valid": False,
                "medical_accuracy": "unknown",
                "issues": [f"Validation error: {str(e)}"],
                "terminology_correct": False,
                "validation_successful": False
            }
            debugger.log_error(
                "Erreur lors de la validation médicale",
                "Error during medical validation",
                e
            )

        return validation_result

    def _check_factual_alignment(self, response: str, passages: List) -> Dict[str, Any]:
        """
        Verify factual alignment with source passages.
        
        Args:
            response: Response to verify
            passages: Source passages for verification
            
        Returns:
            Dict with factual alignment results
        """
        if not passages:
            return {"valid": False, "reason": "No source passages"}

        # Use existing grader for each relevant passage
        alignment_scores = []
        for i, (doc, score) in enumerate(passages[:5]):  # Limit to first 5
            try:
                grader_result = retrieval_grader(response, doc.page_content)
                alignment_scores.append(grader_result.get('score', 0))
            except Exception as e:
                debugger.log_warning(
                    f"Erreur du grader pour le passage {i}",
                    f"Grader error for passage {i}"
                )
                alignment_scores.append(0)

        avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        result = {
            "valid": avg_alignment >= 0.6,
            "alignment_score": avg_alignment,
            "aligned_passages": sum(1 for s in alignment_scores if s >= 0.7),
            "total_checked": len(alignment_scores)
        }
        
        debugger.log_debug(
            f"Alignement factuel: {avg_alignment:.2f} score moyen",
            f"Factual alignment: {avg_alignment:.2f} average score",
            result
        )
        
        return result

    def _check_completeness(self, response: str, question: str) -> Dict[str, Any]:
        """
        Evaluate if the response is complete relative to the question.
        
        Args:
            response: Generated response
            question: Original question
            
        Returns:
            Dict with completeness assessment
        """
        # Keywords for different types of medical questions
        diagnostic_keywords = ["diagnostic", "diagnose", "maladie", "pathologie", "symptôme"]
        treatment_keywords = ["traitement", "thérapie", "médicament", "posologie", "prescription"]
        
        question_lower = question.lower()
        response_lower = response.lower()
        
        completeness_score = 0.5  # Base score
        
        # Check if question asks for diagnosis
        if any(keyword in question_lower for keyword in diagnostic_keywords):
            if any(keyword in response_lower for keyword in ["diagnostic", "diagnose", "pathologie"]):
                completeness_score += 0.3
        
        # Check if question asks for treatment
        if any(keyword in question_lower for keyword in treatment_keywords):
            if any(keyword in response_lower for keyword in treatment_keywords):
                completeness_score += 0.3
        
        # Penalty for too short or vague responses
        if len(response.split()) < 10:
            completeness_score -= 0.2
        
        if "information insuffisante" in response_lower:
            completeness_score -= 0.1
        
        result = {
            "score": min(1.0, max(0.0, completeness_score)),
            "word_count": len(response.split()),
            "addresses_diagnostic": any(k in response_lower for k in diagnostic_keywords),
            "addresses_treatment": any(k in response_lower for k in treatment_keywords)
        }
        
        debugger.log_debug(
            f"Score de complétude: {result['score']:.2f}",
            f"Completeness score: {result['score']:.2f}",
            result
        )
        
        return result

    def _calculate_confidence_level(self, validation_results: Dict[str, Any]) -> str:
        """
        Calculate global confidence level based on validation results.
        
        Args:
            validation_results: All validation results
            
        Returns:
            Confidence level string ('high', 'medium', 'low')
        """
        score = 0
        max_score = 4

        # Citation accuracy (25%)
        if validation_results["citation_accuracy"]["valid"]:
            score += 1

        # Medical consistency (25%)
        if validation_results["medical_consistency"]["valid"]:
            score += 1

        # Factual alignment (25%)
        if validation_results["factual_alignment"]["valid"]:
            score += 1

        # Completeness (25%)
        if validation_results["completeness_score"]["score"] >= 0.7:
            score += 1

        confidence_ratio = score / max_score

        if confidence_ratio >= 0.8:
            return "high"
        elif confidence_ratio >= 0.6:
            return "medium"
        else:
            return "low"

    def generate_improvement_suggestions(self, validation_results: Dict[str, Any]) -> List[str]:
        """
        Generate improvement suggestions based on validation results.
        
        Args:
            validation_results: Validation results to analyze
            
        Returns:
            List of improvement suggestions in French
        """
        suggestions = []
        citation_results = validation_results["citation_accuracy"]
        
        if not citation_results["valid"]:
            if citation_results["uncited_sentences"]:
                suggestions.append("Ajouter des citations pour les affirmations factuelles")
            if citation_results["invalid_citations"]:
                suggestions.append("Corriger les numéros de citation invalides")
        
        if not validation_results["medical_consistency"]["valid"]:
            suggestions.append("Réviser la cohérence médicale de la réponse")
        
        if not validation_results["factual_alignment"]["valid"]:
            suggestions.append("Mieux aligner la réponse avec les documents source")
        
        if validation_results["completeness_score"]["score"] < 0.7:
            suggestions.append("Enrichir la réponse pour mieux répondre à la question")
        
        debugger.log_debug(
            f"Suggestions générées: {len(suggestions)}",
            f"Suggestions generated: {len(suggestions)}",
            suggestions
        )
        
        return suggestions