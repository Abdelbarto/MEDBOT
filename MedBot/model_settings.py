# -*- coding: utf-8 -*-
"""
Advanced configuration management with predefined profiles.

This module provides comprehensive configuration management for the Medical RAG
system with multiple predefined profiles optimized for different use cases.

Features:
- Multiple performance profiles (performance, accuracy, balanced)
- Dynamic profile switching
- Configuration persistence and loading
- Parameter validation and constraints
- Runtime configuration updates

Author: Souleiman & Abdelbar Medical RAG System
Created: 2025
"""

import json
from typing import Dict, Any
from debugging import get_debugger, debug_decorator

# Initialize debugger
debugger = get_debugger()


class Model_Settings:
    """
    Advanced configuration with predefined profiles for different use cases.
    
    Available profiles: 'performance', 'accuracy', 'balanced'
    Each profile is optimized for specific requirements balancing speed vs. precision.
    """

    def __init__(self, profile: str = "balanced", custom_config: Dict[str, Any] = None):
        """
        Initialize settings with specified profile and optional custom overrides.
        
        Args:
            profile: Configuration profile to use ('performance', 'accuracy', 'balanced')
            custom_config: Optional dictionary of custom configuration overrides
        """
        self.profile = profile
        self.custom_config = custom_config or {}
        
        self._load_profile_settings()
        self._apply_custom_overrides()
        
        debugger.log_info(
            f"Configuration initialisée avec le profil: {profile}",
            f"Configuration initialized with profile: {profile}"
        )

    @debug_decorator(debugger, "load_profile_settings",
                    "Chargement des paramètres de profil",
                    "Loading profile settings")
    def _load_profile_settings(self):
        """Load predefined profile settings optimized for different use cases."""
        profiles = {
            "performance": {
                # Optimized for speed
                "MODEL_TYPE": "Ollama",
                "MODEL_NAME": "llama3.2:1b",
                "NUM_PREDICT": 2000,
                "TEMPERATURE": 0.1,
                "TOP_K": 40,
                "TOP_P": 0.85,
                "REPEAT_PENALTY": 1.05,
                "RETRIEVAL_TOP_K": 8,
                "RETRIEVAL_THRESHOLD": 0.15,
                "IS_RETRIEVAL": True,
                "IS_GRADER": False,
                "CACHE_ENABLED": True,
                "VALIDATION_ENABLED": False,
                "MONITORING_ENABLED": False
            },
            "accuracy": {
                # Optimized for maximum precision
                "MODEL_TYPE": "Ollama",
                "MODEL_NAME": "llama3.2:3b",
                "NUM_PREDICT": 4000,
                "TEMPERATURE": 0.0,
                "TOP_K": 80,
                "TOP_P": 0.95,
                "REPEAT_PENALTY": 1.15,
                "RETRIEVAL_TOP_K": 25,
                "RETRIEVAL_THRESHOLD": 0.05,
                "IS_RETRIEVAL": True,
                "IS_GRADER": True,
                "CACHE_ENABLED": True,
                "VALIDATION_ENABLED": True,
                "MONITORING_ENABLED": True,
                "CITATION_REQUIRED": True,
                "DOUBLE_CHECK": True
            },
            "balanced": {
                # Balance between performance and precision
                "MODEL_TYPE": "Ollama",
                "MODEL_NAME": "llama3.2:1b",
                "NUM_PREDICT": 3000,
                "TEMPERATURE": 0.2,
                "TOP_K": 60,
                "TOP_P": 0.95,
                "REPEAT_PENALTY": 1.12,
                "RETRIEVAL_TOP_K": 15,
                "RETRIEVAL_THRESHOLD": 0.08,
                "IS_RETRIEVAL": True,
                "IS_GRADER": True,
                "CACHE_ENABLED": True,
                "VALIDATION_ENABLED": False,
                "MONITORING_ENABLED": True,
                "CITATION_REQUIRED": False
            }
        }

        selected_profile = profiles.get(self.profile, profiles["balanced"])
        
        # Apply all profile settings as instance attributes
        for key, value in selected_profile.items():
            setattr(self, key, value)
            
        debugger.log_debug(
            f"Paramètres de profil '{self.profile}' chargés",
            f"Profile '{self.profile}' settings loaded",
            {"profile": self.profile, "settings_count": len(selected_profile)}
        )

    def _apply_custom_overrides(self):
        """
        Apply custom configuration overrides.
        
        Custom overrides take precedence over profile defaults.
        """
        if not self.custom_config:
            return
            
        overridden_keys = []
        for key, value in self.custom_config.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                overridden_keys.append(f"{key}: {old_value} → {value}")
            else:
                setattr(self, key, value)
                overridden_keys.append(f"{key}: NEW → {value}")
        
        if overridden_keys:
            debugger.log_info(
                f"Overrides appliqués: {len(overridden_keys)} paramètres",
                f"Overrides applied: {len(overridden_keys)} parameters",
                {"overrides": overridden_keys}
            )

    @debug_decorator(debugger, "switch_profile",
                    "Changement de profil de configuration",
                    "Switch configuration profile")
    def switch_profile(self, new_profile: str):
        """
        Switch to a different configuration profile at runtime.
        
        Args:
            new_profile: New profile name to switch to
        """
        old_profile = self.profile
        self.profile = new_profile
        
        # Reload settings with new profile
        self._load_profile_settings()
        self._apply_custom_overrides()
        
        debugger.log_info(
            f"Profil changé: {old_profile} → {new_profile}",
            f"Profile switched: {old_profile} → {new_profile}"
        )

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Return current configuration as dictionary.
        
        Returns:
            Dict containing all configuration attributes
        """
        attrs = [
            attr for attr in dir(self) 
            if not attr.startswith('_') and not callable(getattr(self, attr))
        ]
        return {attr: getattr(self, attr) for attr in attrs}

    def save_config(self, filename: str):
        """
        Save current configuration to file.
        
        Args:
            filename: File path to save configuration to
        """
        config = self.get_config_dict()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            debugger.log_info(
                f"Configuration sauvegardée dans {filename}",
                f"Configuration saved to {filename}"
            )
            
        except Exception as e:
            debugger.log_error(
                f"Erreur de sauvegarde de configuration: {filename}",
                f"Configuration save error: {filename}",
                e
            )
            raise

    def load_config(self, filename: str):
        """
        Load configuration from file.
        
        Args:
            filename: File path to load configuration from
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            loaded_keys = []
            for key, value in config.items():
                if not key.startswith('_'):  # Skip private attributes
                    setattr(self, key, value)
                    loaded_keys.append(key)
            
            debugger.log_info(
                f"Configuration chargée depuis {filename}: {len(loaded_keys)} paramètres",
                f"Configuration loaded from {filename}: {len(loaded_keys)} parameters",
                {"loaded_keys": loaded_keys}
            )
            
        except Exception as e:
            debugger.log_error(
                f"Erreur de chargement de configuration: {filename}",
                f"Configuration load error: {filename}",
                e
            )
            raise

    def validate_settings(self) -> Dict[str, Any]:
        """
        Validate current settings and return validation results.
        
        Returns:
            Dict with validation results and any issues found
        """
        issues = []
        warnings = []
        
        # Temperature validation
        if hasattr(self, 'TEMPERATURE'):
            if not (0.0 <= self.TEMPERATURE <= 2.0):
                issues.append(f"TEMPERATURE {self.TEMPERATURE} outside valid range [0.0, 2.0]")
        
        # TOP_P validation
        if hasattr(self, 'TOP_P'):
            if not (0.0 <= self.TOP_P <= 1.0):
                issues.append(f"TOP_P {self.TOP_P} outside valid range [0.0, 1.0]")
        
        # TOP_K validation
        if hasattr(self, 'TOP_K'):
            if not (1 <= self.TOP_K <= 100):
                warnings.append(f"TOP_K {self.TOP_K} outside recommended range [1, 100]")
        
        # Retrieval settings validation
        if hasattr(self, 'RETRIEVAL_TOP_K'):
            if not (1 <= self.RETRIEVAL_TOP_K <= 50):
                warnings.append(f"RETRIEVAL_TOP_K {self.RETRIEVAL_TOP_K} outside recommended range [1, 50]")
        
        if hasattr(self, 'RETRIEVAL_THRESHOLD'):
            if not (0.0 <= self.RETRIEVAL_THRESHOLD <= 1.0):
                issues.append(f"RETRIEVAL_THRESHOLD {self.RETRIEVAL_THRESHOLD} outside valid range [0.0, 1.0]")
        
        # Performance vs. accuracy profile consistency
        if hasattr(self, 'IS_GRADER') and hasattr(self, 'VALIDATION_ENABLED'):
            if self.profile == "performance" and (self.IS_GRADER or self.VALIDATION_ENABLED):
                warnings.append("Performance profile with expensive validation enabled")
        
        validation_result = {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "profile": self.profile
        }
        
        if issues or warnings:
            debugger.log_warning(
                f"Validation de configuration: {len(issues)} erreurs, {len(warnings)} avertissements",
                f"Configuration validation: {len(issues)} errors, {len(warnings)} warnings",
                validation_result
            )
        else:
            debugger.log_info(
                "Configuration validée avec succès",
                "Configuration validated successfully"
            )
        
        return validation_result

    def get_profile_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Return comparison of all available profiles.
        
        Returns:
            Dict mapping profile names to their settings
        """
        original_profile = self.profile
        original_custom = self.custom_config
        
        comparison = {}
        
        for profile_name in ["performance", "accuracy", "balanced"]:
            # Temporarily switch to each profile
            self.profile = profile_name
            self.custom_config = {}
            self._load_profile_settings()
            
            # Get key settings for comparison
            comparison[profile_name] = {
                "MODEL_NAME": getattr(self, 'MODEL_NAME', 'Unknown'),
                "TEMPERATURE": getattr(self, 'TEMPERATURE', 0.0),
                "RETRIEVAL_TOP_K": getattr(self, 'RETRIEVAL_TOP_K', 0),
                "RETRIEVAL_THRESHOLD": getattr(self, 'RETRIEVAL_THRESHOLD', 0.0),
                "IS_GRADER": getattr(self, 'IS_GRADER', False),
                "VALIDATION_ENABLED": getattr(self, 'VALIDATION_ENABLED', False),
                "MONITORING_ENABLED": getattr(self, 'MONITORING_ENABLED', False)
            }
        
        # Restore original settings
        self.profile = original_profile
        self.custom_config = original_custom
        self._load_profile_settings()
        self._apply_custom_overrides()
        
        return comparison

    def __repr__(self):
        """
        Return string representation of configuration.
        
        Returns:
            Formatted string showing key configuration parameters
        """
        attrs = [
            'profile', 'MODEL_TYPE', 'MODEL_NAME', 'NUM_PREDICT',
            'TEMPERATURE', 'TOP_K', 'TOP_P', 'REPEAT_PENALTY',
            'RETRIEVAL_TOP_K', 'RETRIEVAL_THRESHOLD', 'IS_RETRIEVAL', 'IS_GRADER'
        ]

        lines = []
        for name in attrs:
            if hasattr(self, name):
                value = getattr(self, name)
                val_str = f"'{value}'" if isinstance(value, str) else repr(value)
                lines.append(f"  {name:<20} = {val_str}")

        return f"<{self.__class__.__name__}(\n" + ",\n".join(lines) + "\n)>"

    def __str__(self):
        """
        Return human-readable string representation.
        
        Returns:
            Formatted string with key settings
        """
        return (
            f"Model Settings (Profile: {self.profile})\n"
            f"  Model: {getattr(self, 'MODEL_NAME', 'Unknown')}\n"
            f"  Temperature: {getattr(self, 'TEMPERATURE', 'Unknown')}\n"
            f"  Retrieval K: {getattr(self, 'RETRIEVAL_TOP_K', 'Unknown')}\n"
            f"  Grader: {getattr(self, 'IS_GRADER', 'Unknown')}\n"
            f"  Validation: {getattr(self, 'VALIDATION_ENABLED', 'Unknown')}"
        )