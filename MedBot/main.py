# -*- coding: utf-8 -*-
"""
Medical RAG System - Main application entry point.

This module provides the main interface for the Medical RAG system,
including interactive chat, configuration management, and document handling.

Features:
- Interactive medical question answering
- Document management (permanent/temporary)
- Performance monitoring and optimization
- Advanced query syntax support
- Configuration profiles management

Author: Souleiman & Abdelbar Medical RAG System
Created: 2025
"""

import time
import os
import atexit
import sys
from typing import Dict, Any

from llm_client import OllamaClient
from RAGSystem import RAGSystem
from debugging import get_debugger, debug_decorator

# Initialize debugger
debugger = get_debugger()


@debug_decorator(debugger, "edit_retrieval_settings",
                "Configuration des paramètres de récupération",
                "Configure retrieval parameters")
def edit_retrieval(settings):
    """
    Prompt user to adjust retrieval parameters.
    
    Args:
        settings: Model settings instance to modify
    """
    debugger.log_info("Modification des paramètres de récupération", "Editing retrieval settings")
    
    print(f"\n🔧 Configuration Récupération Actuelle:")
    print(f" TOP_K: {settings.RETRIEVAL_TOP_K}")
    print(f" THRESHOLD: {settings.RETRIEVAL_THRESHOLD}")
    print(f" GRADER: {settings.IS_GRADER}")

    k = input(f"\nNombre de passages à récupérer [{settings.RETRIEVAL_TOP_K}]: ").strip()
    thr = input(f"Seuil de similarité [{settings.RETRIEVAL_THRESHOLD}]: ").strip()
    grader = input(f"Activer le grader? (o/n) [{'o' if settings.IS_GRADER else 'n'}]: ").strip()

    if k.isdigit() and 1 <= int(k) <= 50:
        settings.RETRIEVAL_TOP_K = int(k)

    try:
        if thr and 0.0 <= float(thr) <= 1.0:
            settings.RETRIEVAL_THRESHOLD = float(thr)
    except ValueError:
        debugger.log_warning("Seuil invalide, conservation de la valeur actuelle", "Invalid threshold, keeping current value")

    if grader.lower() in ['o', 'oui', 'y', 'yes']:
        settings.IS_GRADER = True
    elif grader.lower() in ['n', 'non', 'no']:
        settings.IS_GRADER = False


@debug_decorator(debugger, "edit_generation_settings",
                "Configuration des paramètres de génération",
                "Configure generation parameters")
def edit_generation(settings):
    """
    Prompt user to adjust LLM generation parameters.
    
    Args:
        settings: Model settings instance to modify
    """
    debugger.log_info("Modification des paramètres de génération", "Editing generation settings")
    
    print(f"\n🔧 Paramètres de Génération:")
    params = {
        "TEMPERATURE": ("Température (créativité)", settings.TEMPERATURE, 0.0, 2.0),
        "TOP_K": ("Top-K (diversité)", settings.TOP_K, 1, 100),
        "TOP_P": ("Top-P (nucleus)", settings.TOP_P, 0.0, 1.0),
        "REPEAT_PENALTY": ("Pénalité répétition", settings.REPEAT_PENALTY, 1.0, 2.0),
        "NUM_PREDICT": ("Tokens max", settings.NUM_PREDICT, 100, 8192)
    }

    for attr, (description, current, min_val, max_val) in params.items():
        val = input(f"{description} [{current}]: ").strip()
        if val:
            try:
                parsed = float(val) if isinstance(current, float) else int(val)
                if min_val <= parsed <= max_val:
                    setattr(settings, attr, parsed)
                else:
                    debugger.log_warning(f"Valeur hors limites [{min_val}-{max_val}]", f"Value out of range [{min_val}-{max_val}]")
            except (ValueError, TypeError):
                debugger.log_warning(f"Valeur invalide pour {attr}", f"Invalid value for {attr}")


@debug_decorator(debugger, "edit_chunking_settings",
                "Configuration des paramètres de découpage",
                "Configure chunking parameters")
def edit_chunking(memory):
    """
    Prompt user to adjust document chunking parameters.
    
    Args:
        memory: MemoryBuilder instance to modify
    """
    debugger.log_info("Modification des paramètres de découpage", "Editing chunking settings")
    
    print(f"\n🔧 Paramètres de Découpage:")
    params = {
        "chunk_size": ("Taille chunk", memory.chunk_size, 512, 8192),
        "chunk_overlap": ("Chevauchement chunk", memory.chunk_overlap, 0, 500),
        "parent_chunk_size": ("Taille parent", memory.parent_chunk_size, 1024, 16384),
        "parent_chunk_overlap": ("Chevauchement parent", memory.parent_chunk_overlap, 0, 1000)
    }

    changes_made = False
    for name, (description, current, min_val, max_val) in params.items():
        val = input(f"{description} [{current}]: ").strip()
        if val and val.isdigit():
            new_val = int(val)
            if min_val <= new_val <= max_val:
                setattr(memory, name, new_val)
                changes_made = True
            else:
                debugger.log_warning(f"Valeur hors limites [{min_val}-{max_val}]", f"Value out of range [{min_val}-{max_val}]")

    if changes_made:
        debugger.log_info("Réinstanciation du memory builder", "Reinstantiating memory builder")
        memory.instantiate()


@debug_decorator(debugger, "edit_models_settings",
                "Configuration des modèles",
                "Configure models")
def edit_models(settings, rag):
    """
    Prompt user to switch embedding model or LLM.
    
    Args:
        settings: Model settings instance
        rag: RAGSystem instance
    """
    debugger.log_info("Modification des modèles", "Editing models")
    
    print(f"\n🤖 Modèles Disponibles:")
    available_models = [
        "llama3.2:1b",
        "llama3.2:3b",
        "deepseek-r1:1.5b",
        "jpacifico/french-alpaca-3b:latest",
        "qwen2.5:3b"
    ]

    for i, model in enumerate(available_models, 1):
        marker = " (actuel)" if model == settings.MODEL_NAME else ""
        print(f" {i}. {model}{marker}")

    choice = input(f"\nSélection (1-{len(available_models)}) ou nom custom: ").strip()
    new_model = None

    if choice.isdigit() and 1 <= int(choice) <= len(available_models):
        new_model = available_models[int(choice) - 1]
    elif choice:
        new_model = choice

    if new_model and new_model != settings.MODEL_NAME:
        debugger.log_info(f"Changement de modèle: {settings.MODEL_NAME} → {new_model}",
                         f"Model change: {settings.MODEL_NAME} → {new_model}")
        
        settings.MODEL_NAME = new_model

        # Test new model
        debugger.log_info("Test de connexion du nouveau modèle", "Testing new model connection")
        try:
            test_client = OllamaClient(model_name=new_model)
            test_result = test_client.test_connection()
            
            if test_result["success"]:
                rag.llm = test_client
                debugger.log_info("Modèle changé avec succès", "Model changed successfully")
            else:
                debugger.log_error(f"Échec du test: {test_result.get('error', 'Erreur inconnue')}",
                                 f"Test failed: {test_result.get('error', 'Unknown error')}")
                debugger.log_info("Conservation du modèle précédent", "Keeping previous model")
                
        except Exception as e:
            debugger.log_error("Erreur lors du changement de modèle", "Error during model change", e)


def manage_folders(rag):
    """
    Display menu to manage document folders.
    
    Args:
        rag: RAGSystem instance
    """
    memory = rag.memory

    while True:
        print("\n📁 Gestion des Documents:")
        print("1. Charger documents permanents")
        print("2. Charger documents temporaires")
        print("3. Charger tous les documents")
        print("4. Lister sources permanentes")
        print("5. Lister sources temporaires")
        print("6. Lister toutes les sources")
        print("7. Nettoyer documents temporaires")
        print("8. Statistiques cache")
        print("9. Retour")

        choice = input("\nChoix: ").strip()

        if choice == "1":
            rag.load_documents('uploaded_docs/permanent')
        elif choice == "2":
            rag.load_documents('uploaded_docs/temporary')
        elif choice == "3":
            rag.load_all_documents()
        elif choice == "4":
            perm_sources = memory.list_sources_by_type("permanent")
            print(f"\n🔒 Sources Permanentes ({len(perm_sources)}):")
            for src in perm_sources:
                print(f" • {src}")
        elif choice == "5":
            temp_sources = memory.list_sources_by_type("temporary")
            print(f"\n⏰ Sources Temporaires ({len(temp_sources)}):")
            for src in temp_sources:
                print(f" • {src}")
        elif choice == "6":
            memory.list_sources_in_db(repr=True)
        elif choice == "7":
            confirm = input("Nettoyer tous les documents temporaires? (o/N): ").strip().lower()
            if confirm in ['o', 'oui']:
                memory.clear_temporary()
        elif choice == "8":
            if hasattr(memory, 'get_cache_stats'):
                cache_stats = memory.get_cache_stats()
                print(f"\n💾 Statistiques Cache:")
                for key, value in cache_stats.items():
                    print(f" • {key}: {value}")
            else:
                print("Cache non disponible")
        else:
            break


def performance_menu(rag):
    """
    Performance management menu.
    
    Args:
        rag: RAGSystem instance
    """
    while True:
        print("\n⚡ Gestion Performance:")
        print("1. Statut système")
        print("2. Statistiques LLM")
        print("3. Statistiques Grader")
        print("4. Optimisation automatique")
        print("5. Nettoyer cache")
        print("6. Test de performance")
        print("7. Rapport de débogage")
        print("8. Retour")

        choice = input("\nChoix: ").strip()

        if choice == "1":
            status = rag.get_system_status()
            print(f"\n📊 Statut Système:")
            print(f" • Uptime: {status['uptime_seconds']:.1f}s")
            print(f" • Requêtes totales: {status['session_stats']['total_queries']}")
            print(f" • Temps moyen: {status['session_stats']['avg_response_time']:.3f}s")
            print(f" • Sources: {status['memory']['total_sources']}")
            
            if 'quality' in status:
                quality = status['quality']
                print(f" • Score qualité: {quality.get('overall_score', {}).get('score', 'N/A')}")
                print(f" • Santé système: {quality.get('system_health', 'UNKNOWN')}")
                
        elif choice == "2":
            if hasattr(rag.llm, 'get_detailed_stats'):
                stats = rag.llm.get_detailed_stats()
                print(f"\n🤖 Statistiques LLM:")
                for key, value in stats.items():
                    if key != 'config':
                        print(f" • {key}: {value}")
                        
        elif choice == "3":
            try:
                from grader import get_grader_stats
                stats = get_grader_stats()
                print(f"\n🎯 Statistiques Grader:")
                for key, value in stats.items():
                    print(f" • {key}: {value}")
            except ImportError:
                debugger.log_warning("Module grader non disponible", "Grader module not available")
                
        elif choice == "4":
            debugger.log_info("Optimisation automatique en cours", "Running automatic optimization")
            result = rag.optimize_performance()
            print(f"✅ Optimisations: {', '.join(result['optimizations_performed']) if result['optimizations_performed'] else 'Aucune nécessaire'}")
            
        elif choice == "5":
            if hasattr(rag.memory, 'clean_expired_cache'):
                rag.memory.clean_expired_cache()
                debugger.log_info("Cache nettoyé", "Cache cleaned")
                
            try:
                from grader import clear_grader_cache
                clear_grader_cache()
                debugger.log_info("Cache grader nettoyé", "Grader cache cleaned")
            except:
                pass
                
        elif choice == "6":
            debugger.log_info("Test de performance en cours", "Running performance test")
            test_question = "Qu'est-ce que l'hypertension?"
            
            start_time = time.time()
            result = rag.ask_question(question=test_question, debug=True)
            end_time = time.time()
            
            debugger.log_info(f"Test terminé en {end_time - start_time:.3f}s", f"Test completed in {end_time - start_time:.3f}s")
            
            if 'performance' in result:
                perf = result['performance']
                print(f" • Récupération: {perf.get('retrieval_time', 0):.3f}s")
                print(f" • Génération: {perf.get('generation_time', 0):.3f}s")
                
        elif choice == "7":
            debugger.print_performance_report()
            export_file = debugger.export_debug_data()
            print(f"📄 Données exportées: {export_file}")
            
        else:
            break


def advanced_query_menu(rag):
    """
    Advanced queries menu.
    
    Args:
        rag: RAGSystem instance
    """
    print("\n🔍 Requêtes Avancées:")
    print("Syntaxe spéciale:")
    print(" • doc:filename.pdf votre question")
    print(" • type:permanent votre question")
    print(" • type:temporary votre question")
    print(" • debug: votre question (mode debug)")
    print(" • nocache: votre question (bypass cache)")

    while True:
        query = input("\n❓ Requête avancée (ou 'retour'): ").strip()
        if not query or query.lower() == 'retour':
            break

        # Query parsing
        where = None
        debug = False
        bypass_cache = False
        question = query

        if query.startswith("doc:"):
            parts = query.split(maxsplit=1)
            if len(parts) == 2:
                doc_name = parts[0][4:]  # Remove "doc:"
                question = parts[1]
                where = {"source": doc_name}
                
        elif query.startswith("type:"):
            parts = query.split(maxsplit=1)
            if len(parts) == 2:
                doc_type = parts[0][5:]  # Remove "type:"
                question = parts[1]
                where = {"doc_type": doc_type}
                
        elif query.startswith("debug:"):
            question = query[6:].strip()
            debug = True
            
        elif query.startswith("nocache:"):
            question = query[8:].strip()
            bypass_cache = True

        if not question:
            debugger.log_warning("Question vide", "Empty question")
            continue

        # Execute query
        start = time.time()
        result = rag.ask_question(
            question=question,
            where=where,
            debug=debug,
            bypass_cache=bypass_cache
        )
        end = time.time()

        # Display results
        rag.display_result(result, show_performance=True, show_validation=True)
        print(f"\n⌛ Temps total: {end - start:.3f}s")

        # Display applied filters
        if where:
            print(f"🎯 Filtres: {where}")


def change_profile(rag):
    """
    Change configuration profile.
    
    Args:
        rag: RAGSystem instance
    """
    print(f"\n📋 Profils Disponibles:")
    profiles = {
        "1": ("performance", "Optimisé pour la vitesse"),
        "2": ("accuracy", "Optimisé pour la précision"),
        "3": ("balanced", "Équilibré vitesse/précision")
    }

    current_profile = rag.settings.profile
    for key, (name, desc) in profiles.items():
        marker = " (actuel)" if name == current_profile else ""
        print(f" {key}. {name}: {desc}{marker}")

    choice = input(f"\nSélection (1-3): ").strip()
    
    if choice in profiles:
        new_profile = profiles[choice][0]
        if new_profile != current_profile:
            debugger.log_info(f"Changement de profil: {current_profile} → {new_profile}",
                             f"Profile change: {current_profile} → {new_profile}")
            rag.settings.switch_profile(new_profile)


def settings_menu(rag):
    """
    Main settings menu.
    
    Args:
        rag: RAGSystem instance
    """
    while True:
        print(f"\n⚙️ Menu Configuration:")
        print(f"Profil actuel: {rag.settings.profile}")
        print("1. Récupération 2. Génération 3. Découpage")
        print("4. Modèles 5. Documents 6. Performance")
        print("7. Requêtes avancées 8. Profil 9. Retour")

        choice = input("\nChoix: ").strip()

        if choice == "1":
            edit_retrieval(rag.settings)
        elif choice == "2":
            edit_generation(rag.settings)
        elif choice == "3":
            edit_chunking(rag.memory)
        elif choice == "4":
            edit_models(rag.settings, rag)
        elif choice == "5":
            manage_folders(rag)
        elif choice == "6":
            performance_menu(rag)
        elif choice == "7":
            advanced_query_menu(rag)
        elif choice == "8":
            change_profile(rag)
        else:
            break

    debugger.log_info("Configuration mise à jour", "Configuration updated")


def parse_query(raw: str):
    """
    Parse raw input with improved filtering and debug options.
    
    Args:
        raw: Raw user input
        
    Returns:
        Tuple of (question, where_filter, debug_mode)
    """
    where = None
    question = raw
    debug_mode = False

    if raw.startswith("doc:"):
        parts = raw.split(maxsplit=1)
        if len(parts) == 2:
            doc_name = parts[0][4:]  # Remove "doc:"
            question = parts[1]
            where = {"source": doc_name}
            debugger.log_info(f"Document filter applied: looking for '{doc_name}'",
                             f"Document filter applied: looking for '{doc_name}'")
            
    elif raw.startswith("type:"):
        parts = raw.split(maxsplit=1)
        if len(parts) == 2:
            doc_type = parts[0][5:]  # Remove "type:"
            question = parts[1]
            where = {"doc_type": doc_type}
            debugger.log_info(f"Type filter applied: looking for '{doc_type}' documents",
                             f"Type filter applied: looking for '{doc_type}' documents")
            
    elif raw.startswith("debug:"):
        question = raw[6:].strip()
        debug_mode = True
        debugger.log_info("Mode debug activé", "Debug mode activated")
        
    elif raw.startswith("doc-debug:"):
        parts = raw.split(maxsplit=1)
        if len(parts) == 2:
            doc_name = parts[0][10:]  # Remove "doc-debug:"
            question = parts[1]
            where = {"source": doc_name}
            debug_mode = True
            debugger.log_info(f"Document filter + debug: looking for '{doc_name}'",
                             f"Document filter + debug: looking for '{doc_name}'")

    return question, where, debug_mode


def ensure_folder_structure():
    """Create required folder structure if it doesn't exist."""
    folders = [
        'uploaded_docs/permanent',
        'uploaded_docs/temporary',
        'logs',
        'cache'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    debugger.log_info("Structure de dossiers créée", "Folder structure created")


@debug_decorator(debugger, "cleanup_on_exit",
                "Nettoyage à la sortie",
                "Exit cleanup")
def cleanup_on_exit(rag_system):
    """
    Cleanup function called when program exits.
    
    Args:
        rag_system: RAGSystem instance to cleanup
    """
    try:
        # Cleanup temporary documents
        rag_system.memory.clear_temporary()

        # Display final statistics
        status = rag_system.get_system_status()
        debugger.log_info(
            f"Session terminée - Durée: {status['uptime_seconds']:.1f}s, Requêtes: {status['session_stats']['total_queries']}",
            f"Session ended - Duration: {status['uptime_seconds']:.1f}s, Queries: {status['session_stats']['total_queries']}"
        )

        if status['session_stats']['total_queries'] > 0:
            debugger.log_info(
                f"Temps moyen par requête: {status['session_stats']['avg_response_time']:.3f}s",
                f"Average time per query: {status['session_stats']['avg_response_time']:.3f}s"
            )

    except Exception as e:
        debugger.log_error("Erreur lors du nettoyage", "Error during cleanup", e)


def show_help():
    """Display complete help information."""
    help_text = """
🆘 AIDE MEDRAG - SYSTÈME RAG MÉDICAL

=== SYNTAXE AVANCÉE ===
• doc:filename.pdf question → Recherche dans un fichier spécifique
• type:permanent question → Recherche dans docs permanents seulement
• type:temporary question → Recherche dans docs temporaires seulement

=== COMMANDES SPÉCIALES ===
• help → Cette aide
• settings → Menu de configuration
• status → Statut du système
• quit, exit, bye, q → Quitter

=== PROFILS DE CONFIGURATION ===
• performance → Vitesse maximale, moins de vérifications
• accuracy → Précision maximale, validation complète
• balanced → Équilibre vitesse/précision (défaut)

=== EXEMPLES ===
• Quels sont les symptômes de l'hypertension?
• doc:cardio.pdf traitement de l'infarctus
• type:temporary dernières recommandations
• settings (pour configurer)

=== RACCOURCIS CLAVIER ===
• Ctrl+C → Interruption gracieuse
• Ctrl+D → Quitter (Unix/Linux)
"""
    print(help_text)


@debug_decorator(debugger, "main_application",
                "Application principale",
                "Main application")
def main():
    """
    Main entry point for the optimized medical RAG chatbot.
    Automatically loads all documents and provides complete interface.
    """
    # Startup banner
    print("🚀 MedRAG - Système RAG Médical Haute Performance")
    print("=" * 60)

    # Initialize folder structure
    ensure_folder_structure()

    # Detect optimal profile based on resources
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb > 16:
            default_profile = "accuracy"
        elif memory_gb > 8:
            default_profile = "balanced"
        else:
            default_profile = "performance"
            
        debugger.log_info(f"RAM détectée: {memory_gb:.1f}GB → Profil recommandé: {default_profile}",
                         f"RAM detected: {memory_gb:.1f}GB → Recommended profile: {default_profile}")
    except ImportError:
        default_profile = "balanced"
        debugger.log_warning("psutil non disponible, profil par défaut: balanced",
                           "psutil not available, default profile: balanced")

    # RAG system initialization
    debugger.log_info("Initialisation du système RAG", "Initializing RAG system")
    
    try:
        rag = RAGSystem(profile=default_profile)
    except Exception as e:
        debugger.log_error("Erreur d'initialisation principale", "Main initialization error", e)
        debugger.log_info("Tentative avec profil performance", "Trying with performance profile")
        
        try:
            rag = RAGSystem(profile="performance")
        except Exception as e2:
            debugger.log_error("Échec critique d'initialisation", "Critical initialization failure", e2)
            sys.exit(1)

    # Register cleanup on exit
    atexit.register(cleanup_on_exit, rag)

    # Load documents
    debugger.log_info("Chargement des documents", "Loading documents")
    rag.load_all_documents()

    # Display configuration
    status = rag.get_system_status()
    print(f"\n🏗️ Configuration:")
    print(f" • Profil: {rag.settings.profile}")
    print(f" • Modèle: {rag.settings.MODEL_NAME}")
    print(f" • Sources: {status['memory']['total_sources']} documents")
    print(f" • Cache: {'✅' if getattr(rag.settings, 'CACHE_ENABLED', False) else '❌'}")
    print(f" • Validation: {'✅' if rag.validator else '❌'}")
    print(f" • Monitoring: {'✅' if rag.monitor else '❌'}")
    print(f" • Debugging: ✅")

    # User instructions
    print(f"\n📖 Instructions:")
    print(f" • Tapez votre question médicale")
    print(f" • 'help' pour l'aide complète")
    print(f" • 'settings' pour la configuration")
    print(f" • 'status' pour le statut système")
    print(f" • 'quit' pour quitter")

    # Main loop
    try:
        while True:
            raw_query = input(f"\n❓ Question: ").strip()
            
            if not raw_query:
                continue

            # Special commands
            if raw_query.lower() in ("quit", "exit", "bye", "q"):
                break
            elif raw_query.lower() == "help":
                show_help()
                continue
            elif raw_query.lower() == "settings":
                settings_menu(rag)
                continue
            elif raw_query.lower() == "status":
                status = rag.get_system_status()
                print(f"\n📊 Statut Système:")
                for section, data in status.items():
                    if isinstance(data, dict):
                        print(f" {section}:")
                        for key, value in data.items():
                            print(f"   • {key}: {value}")
                    else:
                        print(f" • {section}: {data}")
                continue

            # Query parsing
            question, where, debug_mode = parse_query(raw_query)

            # Processing time measurement
            start_time = time.time()

            # Question processing
            try:
                result = rag.ask_question(
                    question=question,
                    where=where,
                    debug=False  # Debug mode disabled by default
                )

                # Display results
                rag.display_result(result, show_performance=True, show_validation=True)

                # Processing time
                total_time = time.time() - start_time
                print(f"\n⌛ Traitement: {total_time:.3f}s")

                # Display applied filters
                if where:
                    print(f"🎯 Filtre appliqué: {where}")

            except KeyboardInterrupt:
                debugger.log_warning("Requête interrompue par l'utilisateur", "Query interrupted by user")
                continue
            except Exception as e:
                debugger.log_error("Erreur lors du traitement de la question", "Error processing question", e)
                
                if "debug" in raw_query.lower():
                    import traceback
                    traceback.print_exc()
                continue

    except KeyboardInterrupt:
        debugger.log_info("Interruption par l'utilisateur", "Interrupted by user")
    except Exception as e:
        debugger.log_error("Erreur critique dans la boucle principale", "Critical error in main loop", e)
    finally:
        debugger.log_info("Arrêt du système", "System shutdown")


if __name__ == "__main__":
    main()