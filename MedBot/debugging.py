# -*- coding: utf-8 -*-
"""
Comprehensive debugging and logging system for Medical RAG.

This module provides advanced debugging, logging, and performance monitoring
capabilities for the Medical RAG system with structured logging, error tracking,
and performance analysis.

Features:
- Multilingual logging (French/English)
- Performance timing and monitoring
- Structured debug information
- Function call tracing with decorators
- Error tracking and analysis
- Export capabilities for debugging data
- Memory and resource monitoring

Author: Souleiman & Abdelbar Medical RAG System
Created: 2025
"""

import time
import json
import os
import traceback
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import logging
from contextlib import contextmanager


class MedicalRAGDebugger:
    """
    Advanced debugging and performance monitoring system for Medical RAG.
    
    Provides comprehensive logging, performance tracking, error analysis,
    and debugging capabilities with export functionality.
    """

    def __init__(self, log_level: str = "INFO", log_to_file: bool = True, 
                 log_dir: str = "system_files/logs"):
        """
        Initialize the debugging system.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to files
            log_dir: Directory for log files
        """
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        
        # Performance tracking
        self.performance_data = {
            "function_calls": {},
            "timing_data": {},
            "error_counts": {},
            "session_start": datetime.now(),
            "total_operations": 0
        }
        
        # Debug information storage
        self.debug_info = {
            "errors": [],
            "warnings": [],
            "operations": [],
            "performance_alerts": []
        }
        
        # Setup logging
        self._setup_logging()
        
        self.log_info("Système de débogage initialisé", "Debugging system initialized")

    def _setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, 'medical_rag_debug.log')

        # Configure root logger pour n’écrire QUE dans le fichier
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger('MedicalRAG')


    def log_info(self, french_msg: str, english_msg: str, data: Dict[str, Any] = None):
        """
        Log info level message in both languages.
        
        Args:
            french_msg: Message in French
            english_msg: Message in English  
            data: Optional structured data
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "level": "INFO",
            "french": french_msg,
            "english": english_msg,
            "data": data
        }
        
        self.debug_info["operations"].append(log_entry)
        self.logger.info(f"FR: {french_msg} | EN: {english_msg}")
        
        if data:
            self.logger.info(f"Data: {json.dumps(data, default=str, indent=2)}")

    def log_warning(self, french_msg: str, english_msg: str, data: Dict[str, Any] = None):
        """
        Log warning level message in both languages.
        
        Args:
            french_msg: Warning message in French
            english_msg: Warning message in English
            data: Optional structured data
        """
        timestamp = datetime.now().isoformat()
        
        warning_entry = {
            "timestamp": timestamp,
            "level": "WARNING", 
            "french": french_msg,
            "english": english_msg,
            "data": data
        }
        
        self.debug_info["warnings"].append(warning_entry)
        self.logger.warning(f"FR: {french_msg} | EN: {english_msg}")
        
        if data:
            self.logger.warning(f"Data: {json.dumps(data, default=str, indent=2)}")

    def log_error(self, french_msg: str, english_msg: str, exception: Exception = None, 
                  data: Dict[str, Any] = None):
        """
        Log error level message with exception details.
        
        Args:
            french_msg: Error message in French
            english_msg: Error message in English
            exception: Optional exception object
            data: Optional structured data
        """
        timestamp = datetime.now().isoformat()
        
        error_entry = {
            "timestamp": timestamp,
            "level": "ERROR",
            "french": french_msg,
            "english": english_msg,
            "exception": str(exception) if exception else None,
            "traceback": traceback.format_exc() if exception else None,
            "data": data
        }
        
        self.debug_info["errors"].append(error_entry)
        self.logger.error(f"FR: {french_msg} | EN: {english_msg}")
        
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
        if data:
            self.logger.error(f"Data: {json.dumps(data, default=str, indent=2)}")

    def log_debug(self, french_msg: str, english_msg: str, data: Dict[str, Any] = None):
        """
        Log debug level message with detailed data.
        
        Args:
            french_msg: Debug message in French
            english_msg: Debug message in English
            data: Optional structured debugging data
        """
        if self.log_level.upper() != "DEBUG":
            return
            
        timestamp = datetime.now().isoformat()
        
        debug_entry = {
            "timestamp": timestamp,
            "level": "DEBUG",
            "french": french_msg,
            "english": english_msg,
            "data": data
        }
        
        self.debug_info["operations"].append(debug_entry)
        self.logger.debug(f"FR: {french_msg} | EN: {english_msg}")
        
        if data:
            self.logger.debug(f"Debug Data: {json.dumps(data, default=str, indent=2)}")

    @contextmanager
    def timing_context(self, operation_name: str, french_desc: str = "", english_desc: str = ""):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            french_desc: French description
            english_desc: English description
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            # Store timing data
            if operation_name not in self.performance_data["timing_data"]:
                self.performance_data["timing_data"][operation_name] = []
            
            self.performance_data["timing_data"][operation_name].append(duration)
            
            # Log if slow
            if duration > 5.0:
                self.debug_info["performance_alerts"].append({
                    "timestamp": datetime.now().isoformat(),
                    "operation": operation_name,
                    "duration": duration,
                    "threshold_exceeded": "slow_operation"
                })
                
                self.log_warning(
                    f"Opération lente détectée: {operation_name} ({duration:.3f}s)",
                    f"Slow operation detected: {operation_name} ({duration:.3f}s)",
                    {"duration": duration, "operation": operation_name}
                )

    def track_function_call(self, func_name: str, args: tuple = (), kwargs: dict = {}):
        """
        Track function call statistics.
        
        Args:
            func_name: Name of the function called
            args: Function arguments
            kwargs: Function keyword arguments
        """
        if func_name not in self.performance_data["function_calls"]:
            self.performance_data["function_calls"][func_name] = {
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "last_called": None
            }
        
        self.performance_data["function_calls"][func_name]["count"] += 1
        self.performance_data["function_calls"][func_name]["last_called"] = datetime.now().isoformat()
        self.performance_data["total_operations"] += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dict with performance statistics and analysis
        """
        uptime = datetime.now() - self.performance_data["session_start"]
        
        # Calculate average times
        timing_summary = {}
        for operation, times in self.performance_data["timing_data"].items():
            timing_summary[operation] = {
                "count": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }
        
        return {
            "session_uptime": str(uptime),
            "total_operations": self.performance_data["total_operations"],
            "function_calls": self.performance_data["function_calls"],
            "timing_summary": timing_summary,
            "error_count": len(self.debug_info["errors"]),
            "warning_count": len(self.debug_info["warnings"]),
            "performance_alerts_count": len(self.debug_info["performance_alerts"])
        }

    def print_performance_report(self):
        """Print formatted performance report to console."""
        summary = self.get_performance_summary()
        
        print("\n" + "="*80)
        print("📊 RAPPORT DE PERFORMANCE / PERFORMANCE REPORT")
        print("="*80)
        
        print(f"⏱️  Durée de session: {summary['session_uptime']}")
        print(f"🔧 Opérations totales: {summary['total_operations']}")
        print(f"❌ Erreurs: {summary['error_count']}")
        print(f"⚠️  Avertissements: {summary['warning_count']}")
        print(f"🐌 Alertes de performance: {summary['performance_alerts_count']}")
        
        if summary["timing_summary"]:
            print(f"\n📈 Résumé des performances par opération:")
            for operation, stats in summary["timing_summary"].items():
                print(f"  • {operation}:")
                print(f"    - Appels: {stats['count']}")
                print(f"    - Temps moyen: {stats['avg_time']:.3f}s")
                print(f"    - Min/Max: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
        
        print("="*80)

    def export_debug_data(self, filename: Optional[str] = None) -> str:
        """
        Export all debug data to JSON file.
        
        Args:
            filename: Optional filename, auto-generated if not provided
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_export_{timestamp}.json"
        
        export_path = os.path.join(self.log_dir, filename)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "performance_data": self.performance_data,
            "debug_info": self.debug_info,
            "performance_summary": self.get_performance_summary()
        }
        
        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=json_serializer, ensure_ascii=False)
            
            self.log_info(
                f"Données de débogage exportées: {export_path}",
                f"Debug data exported: {export_path}"
            )
            
            return export_path
            
        except Exception as e:
            self.log_error(
                f"Erreur d'exportation: {export_path}",
                f"Export error: {export_path}",
                e
            )
            return ""

    def get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent errors within specified timeframe.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent error entries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = []
        for error in self.debug_info["errors"]:
            error_time = datetime.fromisoformat(error["timestamp"])
            if error_time > cutoff_time:
                recent_errors.append(error)
        
        return recent_errors

    def clear_debug_data(self):
        """Clear all accumulated debug data."""
        self.debug_info = {
            "errors": [],
            "warnings": [],
            "operations": [],
            "performance_alerts": []
        }
        
        self.performance_data = {
            "function_calls": {},
            "timing_data": {},
            "error_counts": {},
            "session_start": datetime.now(),
            "total_operations": 0
        }
        
        self.log_info("Données de débogage effacées", "Debug data cleared")


# Global debugger instance
_global_debugger = None


def get_debugger(log_level: str = "INFO", log_to_file: bool = True) -> MedicalRAGDebugger:
    """
    Get or create global debugger instance.
    
    Args:
        log_level: Logging level for new instance
        log_to_file: Whether to log to files
        
    Returns:
        Global debugger instance
    """
    global _global_debugger
    
    if _global_debugger is None:
        _global_debugger = MedicalRAGDebugger(log_level=log_level, log_to_file=log_to_file)
    
    return _global_debugger


def debug_decorator(debugger: MedicalRAGDebugger, operation_name: str, 
                   french_desc: str, english_desc: str):
    """
    Decorator for automatic function debugging and timing.
    
    Args:
        debugger: Debugger instance to use
        operation_name: Name of the operation
        french_desc: French description
        english_desc: English description
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Track function call
            debugger.track_function_call(func.__name__, args, kwargs)
            
            # Time the operation
            with debugger.timing_context(operation_name, french_desc, english_desc):
                try:
                    debugger.log_debug(
                        f"Début: {french_desc}",
                        f"Starting: {english_desc}",
                        {"function": func.__name__, "operation": operation_name}
                    )
                    
                    result = func(*args, **kwargs)
                    
                    debugger.log_debug(
                        f"Terminé: {french_desc}",
                        f"Completed: {english_desc}",
                        {"function": func.__name__, "operation": operation_name}
                    )
                    
                    return result
                    
                except Exception as e:
                    debugger.log_error(
                        f"Erreur dans {french_desc}",
                        f"Error in {english_desc}",
                        e,
                        {"function": func.__name__, "operation": operation_name}
                    )
                    raise
        
        return wrapper
    return decorator


def log_performance_warning(debugger: MedicalRAGDebugger, operation: str, 
                          duration: float, threshold: float = 5.0):
    """
    Log performance warning if operation exceeds threshold.
    
    Args:
        debugger: Debugger instance
        operation: Operation name
        duration: Operation duration
        threshold: Warning threshold in seconds
    """
    if duration > threshold:
        debugger.log_warning(
            f"Performance: {operation} a pris {duration:.3f}s (> {threshold}s)",
            f"Performance: {operation} took {duration:.3f}s (> {threshold}s)",
            {"operation": operation, "duration": duration, "threshold": threshold}
        )


# Convenience functions for common debugging patterns
def debug_function_entry(debugger: MedicalRAGDebugger, func_name: str, 
                        french_msg: str, english_msg: str, **kwargs):
    """Log function entry with parameters."""
    debugger.log_debug(
        f"→ {func_name}: {french_msg}",
        f"→ {func_name}: {english_msg}",
        kwargs
    )


def debug_function_exit(debugger: MedicalRAGDebugger, func_name: str,
                       french_msg: str, english_msg: str, **kwargs):
    """Log function exit with results."""
    debugger.log_debug(
        f"← {func_name}: {french_msg}",
        f"← {func_name}: {english_msg}",
        kwargs
    )