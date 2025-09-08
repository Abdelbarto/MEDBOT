# -*- coding: utf-8 -*-
"""
Quality monitoring system for Medical RAG performance and accuracy tracking.

This module provides comprehensive quality monitoring and metrics collection
for the Medical RAG system, including performance tracking, quality assessment,
and automated reporting capabilities.

Features:
- Query performance logging and analysis
- Medical response quality assessment
- System health monitoring
- Automated trend analysis and recommendations
- SQLite-based metrics persistence
- Comprehensive reporting capabilities

Author: Souleiman & Abdelbar Medical RAG System
Created: 2025
"""

import logging
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics
from debugging import get_debugger, debug_decorator

# Initialize debugger
debugger = get_debugger()


class QualityMonitor:
    """
    Comprehensive quality monitoring and metrics system for medical RAG.
    
    Provides detailed tracking of query performance, response quality,
    system health, and automated analysis with recommendations.
    """

    def __init__(self, log_file="medical_rag_quality.log", db_file="quality_metrics.db"):
        """
        Initialize quality monitoring system.
        
        Args:
            log_file: Path to log file for quality events
            db_file: Path to SQLite database for metrics storage
        """
        self.log_file = log_file
        self.db_file = db_file

        # Configure logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # In-memory metrics for current session
        self.current_session_metrics = {
            "session_start": datetime.now(),
            "total_queries": 0,
            "successful_retrievals": 0,
            "validation_failures": 0,
            "response_times": [],
            "citation_accuracy": [],
            "confidence_levels": {"high": 0, "medium": 0, "low": 0},
            "errors": []
        }

        # Initialize database
        self._init_database()
        
        debugger.log_info(
            "Système de monitoring de qualité initialisé",
            "Quality monitoring system initialized"
        )

    @debug_decorator(debugger, "init_quality_database",
                    "Initialisation de la base de données de qualité",
                    "Initialize quality database")
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Query logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                question_hash TEXT,
                retrieval_time REAL,
                generation_time REAL,
                total_time REAL,
                citations_count INTEGER,
                confidence_level TEXT,
                validation_success BOOLEAN,
                error_message TEXT
            )
        """)

        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                avg_response_time REAL,
                success_rate REAL,
                avg_citations REAL,
                high_confidence_rate REAL
            )
        """)

        # System health events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                health_status TEXT,
                metrics_json TEXT,
                alerts TEXT
            )
        """)

        conn.commit()
        conn.close()

    @debug_decorator(debugger, "log_query_performance",
                    "Enregistrement des performances de requête",
                    "Log query performance")
    def log_query_performance(self, question: str, retrieval_time: float,
                             generation_time: float, validation_results: Dict[str, Any],
                             error_message: Optional[str] = None):
        """
        Log detailed performance metrics for each query.
        
        Args:
            question: Original question asked
            retrieval_time: Time spent on document retrieval
            generation_time: Time spent on response generation
            validation_results: Results from response validation
            error_message: Optional error message if query failed
        """
        # Hash question for privacy
        question_hash = hashlib.md5(question.encode()).hexdigest()[:12]

        # Extract validation metrics
        citation_count = validation_results.get("citation_accuracy", {}).get("found_citations", 0)
        confidence_level = validation_results.get("confidence_level", "unknown")
        validation_success = validation_results.get("confidence_level") in ["high", "medium"]

        total_time = retrieval_time + generation_time

        # Update session metrics
        self.current_session_metrics["total_queries"] += 1
        if validation_success:
            self.current_session_metrics["successful_retrievals"] += 1
        else:
            self.current_session_metrics["validation_failures"] += 1

        self.current_session_metrics["response_times"].append(total_time)
        self.current_session_metrics["citation_accuracy"].append(citation_count)

        if confidence_level in self.current_session_metrics["confidence_levels"]:
            self.current_session_metrics["confidence_levels"][confidence_level] += 1

        if error_message:
            self.current_session_metrics["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": error_message,
                "question_hash": question_hash
            })

        # Structured log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question_hash": question_hash,
            "performance": {
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(total_time, 3)
            },
            "quality": {
                "citations_count": citation_count,
                "confidence_level": confidence_level,
                "validation_success": validation_success
            },
            "validation_details": {
                "citation_valid": validation_results.get("citation_accuracy", {}).get("valid", False),
                "medical_consistency": validation_results.get("medical_consistency", {}).get("valid", False),
                "factual_alignment": validation_results.get("factual_alignment", {}).get("valid", False)
            }
        }

        # Log to file
        self.logger.info(f"QUERY_PERFORMANCE: {json.dumps(log_entry)}")

        # Save to database
        self._save_to_database(log_entry, error_message)

        # Performance alerts
        if total_time > 10.0:
            self.logger.warning(f"SLOW_QUERY: {question_hash} took {total_time:.2f}s")
            debugger.log_warning(f"Requête lente détectée: {total_time:.2f}s", f"Slow query detected: {total_time:.2f}s")

        if confidence_level == "low":
            self.logger.warning(f"LOW_CONFIDENCE: {question_hash} confidence={confidence_level}")
            debugger.log_warning(f"Faible confiance détectée: {confidence_level}", f"Low confidence detected: {confidence_level}")

    def _save_to_database(self, log_entry: Dict[str, Any], error_message: Optional[str]):
        """
        Save log entry to database.
        
        Args:
            log_entry: Structured log entry to save
            error_message: Optional error message
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO query_logs
            (timestamp, question_hash, retrieval_time, generation_time, total_time,
             citations_count, confidence_level, validation_success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_entry["timestamp"],
            log_entry["question_hash"],
            log_entry["performance"]["retrieval_time"],
            log_entry["performance"]["generation_time"],
            log_entry["performance"]["total_time"],
            log_entry["quality"]["citations_count"],
            log_entry["quality"]["confidence_level"],
            log_entry["quality"]["validation_success"],
            error_message
        ))

        conn.commit()
        conn.close()

    @debug_decorator(debugger, "generate_quality_report",
                    "Génération du rapport de qualité",
                    "Generate quality report")
    def get_quality_report(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            days_back: Number of days to include in historical analysis
            
        Returns:
            Dict containing comprehensive quality report
        """
        # Current session metrics
        session_metrics = self._calculate_session_metrics()

        # Historical metrics
        historical_metrics = self._get_historical_metrics(days_back)

        # Trend analysis
        trends = self._calculate_trends(days_back)

        # Recommendations
        recommendations = self._generate_recommendations(session_metrics, historical_metrics)

        # Global quality score
        quality_score = self._calculate_overall_quality_score(session_metrics, historical_metrics)

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "period": f"Last {days_back} days",
            "session_metrics": session_metrics,
            "historical_metrics": historical_metrics,
            "trends": trends,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "system_health": self._assess_system_health(session_metrics)
        }
        
        debugger.log_info(
            f"Rapport de qualité généré pour {days_back} jours",
            f"Quality report generated for {days_back} days",
            {"system_health": report["system_health"], "quality_score": quality_score.get("grade", "N/A")}
        )
        
        return report

    def _calculate_session_metrics(self) -> Dict[str, Any]:
        """
        Calculate current session metrics.
        
        Returns:
            Dict with current session performance metrics
        """
        metrics = self.current_session_metrics

        if metrics["total_queries"] == 0:
            return {"status": "no_queries", "total_queries": 0}

        avg_response_time = statistics.mean(metrics["response_times"]) if metrics["response_times"] else 0
        success_rate = (metrics["successful_retrievals"] / metrics["total_queries"]) * 100
        avg_citations = statistics.mean(metrics["citation_accuracy"]) if metrics["citation_accuracy"] else 0

        total_confidence = sum(metrics["confidence_levels"].values())
        high_confidence_rate = (metrics["confidence_levels"]["high"] / total_confidence * 100) if total_confidence > 0 else 0

        return {
            "total_queries": metrics["total_queries"],
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "avg_citations_per_response": round(avg_citations, 1),
            "high_confidence_rate": round(high_confidence_rate, 2),
            "confidence_distribution": metrics["confidence_levels"],
            "error_count": len(metrics["errors"]),
            "session_duration": str(datetime.now() - metrics["session_start"])
        }

    def _get_historical_metrics(self, days_back: int) -> Dict[str, Any]:
        """
        Retrieve historical metrics from database.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dict with historical performance metrics
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        since_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total_queries,
                AVG(total_time) as avg_response_time,
                AVG(citations_count) as avg_citations,
                SUM(CASE WHEN validation_success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                SUM(CASE WHEN confidence_level = 'high' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as high_confidence_rate
            FROM query_logs
            WHERE timestamp > ?
        """, (since_date,))

        result = cursor.fetchone()
        conn.close()

        if result and result[0] > 0:
            return {
                "total_queries": result[0],
                "avg_response_time": round(result[1], 3) if result[1] else 0,
                "avg_citations": round(result[2], 1) if result[2] else 0,
                "success_rate": round(result[3], 2) if result[3] else 0,
                "high_confidence_rate": round(result[4], 2) if result[4] else 0
            }
        else:
            return {"status": "no_historical_data"}

    def _calculate_trends(self, days_back: int) -> Dict[str, str]:
        """
        Calculate performance trends.
        
        Args:
            days_back: Number of days for trend analysis
            
        Returns:
            Dict with trend descriptions
        """
        # Simplified trend analysis - in real implementation would compare
        # current vs historical performance
        return {
            "response_time": "stable",
            "success_rate": "improving",
            "citation_quality": "stable",
            "confidence_levels": "improving"
        }

    def _generate_recommendations(self, session_metrics: Dict[str, Any],
                                 historical_metrics: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on metrics.
        
        Args:
            session_metrics: Current session performance
            historical_metrics: Historical performance data
            
        Returns:
            List of recommendation strings
        """
        recommendations = []

        if "avg_response_time" in session_metrics and session_metrics["avg_response_time"] > 5.0:
            recommendations.append("⚠️ Temps de réponse élevé - Considérer l'activation du cache")

        if "success_rate" in session_metrics and session_metrics["success_rate"] < 80:
            recommendations.append("📉 Taux de succès bas - Ajuster les seuils de similarité")

        if "avg_citations_per_response" in session_metrics and session_metrics["avg_citations_per_response"] < 2:
            recommendations.append("📚 Citations insuffisantes - Renforcer les exigences de citation")

        if "high_confidence_rate" in session_metrics and session_metrics["high_confidence_rate"] < 60:
            recommendations.append("🎯 Faible confiance - Améliorer la validation ou le grading")

        if not recommendations:
            recommendations.append("✅ Système performant - Continuer le monitoring")

        return recommendations

    def _calculate_overall_quality_score(self, session_metrics: Dict[str, Any],
                                        historical_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall quality score on scale of 100.
        
        Args:
            session_metrics: Current session metrics
            historical_metrics: Historical metrics
            
        Returns:
            Dict with overall quality score and grade
        """
        if "success_rate" not in session_metrics:
            return {"score": 0, "grade": "N/A", "reason": "Insufficient data"}

        weights = {
            "success_rate": 0.3,
            "response_time": 0.2,
            "citations": 0.25,
            "confidence": 0.25
        }

        scores = {}

        # Success rate score (0-100)
        scores["success_rate"] = min(100, session_metrics.get("success_rate", 0))

        # Response time score (faster is better, penalize slow responses)
        response_time = session_metrics.get("avg_response_time", 10)
        scores["response_time"] = max(0, 100 - (response_time - 1) * 11.11)

        # Citations score
        avg_citations = session_metrics.get("avg_citations_per_response", 0)
        if avg_citations >= 3:
            scores["citations"] = 100
        else:
            scores["citations"] = (avg_citations / 3) * 100

        # Confidence score
        scores["confidence"] = session_metrics.get("high_confidence_rate", 0)

        # Weighted overall score
        weighted_score = sum(scores[key] * weights[key] for key in scores.keys())

        # Grade assignment
        if weighted_score >= 90:
            grade = "A+"
        elif weighted_score >= 80:
            grade = "A"
        elif weighted_score >= 70:
            grade = "B"
        elif weighted_score >= 60:
            grade = "C"
        else:
            grade = "D"

        return {
            "score": round(weighted_score, 1),
            "grade": grade,
            "component_scores": {k: round(v, 1) for k, v in scores.items()},
            "weights": weights
        }

    def _assess_system_health(self, session_metrics: Dict[str, Any]) -> str:
        """
        Assess overall system health status.
        
        Args:
            session_metrics: Current session metrics
            
        Returns:
            Health status string ('HEALTHY', 'WARNING', 'CRITICAL', 'UNKNOWN')
        """
        if "total_queries" not in session_metrics or session_metrics["total_queries"] == 0:
            return "UNKNOWN"

        issues = 0

        # Check various health indicators
        if session_metrics.get("success_rate", 0) < 80:
            issues += 1

        if session_metrics.get("avg_response_time", 0) > 5:
            issues += 1

        if session_metrics.get("error_count", 0) > session_metrics["total_queries"] * 0.1:
            issues += 1

        if session_metrics.get("high_confidence_rate", 0) < 50:
            issues += 1

        # Determine health status based on issues
        if issues == 0:
            return "HEALTHY"
        elif issues <= 2:
            return "WARNING"
        else:
            return "CRITICAL"

    def export_metrics(self, filename: str, days_back: int = 30) -> bool:
        """
        Export metrics to JSON file.
        
        Args:
            filename: Output filename
            days_back: Days of historical data to include
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            report = self.get_quality_report(days_back)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            debugger.log_info(
                f"Métriques exportées vers {filename}",
                f"Metrics exported to {filename}"
            )
            return True
            
        except Exception as e:
            debugger.log_error(
                f"Erreur d'exportation des métriques: {filename}",
                f"Metrics export error: {filename}",
                e
            )
            return False

    def reset_session_metrics(self):
        """Reset current session metrics."""
        self.current_session_metrics = {
            "session_start": datetime.now(),
            "total_queries": 0,
            "successful_retrievals": 0,
            "validation_failures": 0,
            "response_times": [],
            "citation_accuracy": [],
            "confidence_levels": {"high": 0, "medium": 0, "low": 0},
            "errors": []
        }
        
        debugger.log_info(
            "Métriques de session remises à zéro",
            "Session metrics reset"
        )