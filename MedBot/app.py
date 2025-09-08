# -*- coding: utf-8 -*-

"""
Flask Backend for Medical RAG System Web Interface
Backend Flask complet pour l'interface web du système Medical RAG.
Fournit des endpoints API REST qui s'intègrent avec le codebase Python existant.

Author: Integration avec le système Medical RAG de Souleiman & Abdelbar
Created: 2025
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import sys
import json
import traceback
import threading
import time
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

# Ajouter le répertoire courant au path Python pour importer vos modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration de l'application Flask
app = Flask(__name__)
CORS(app)  # Activer CORS pour l'interface web

# Configuration des uploads
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploaded_docs'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md', 'docx', 'doc'}

# Variables globales pour le système RAG
rag_system = None
debugger = None
system_start_time = time.time()
MODULES_LOADED = False
VALIDATION_AVAILABLE = False
MONITORING_AVAILABLE = False

# Tentative d'import des modules
try:
    # Import des modules existants
    from RAGSystem import RAGSystem
    from model_settings import Model_Settings
    from memory_builder import MemoryBuilder
    from llm_client import OllamaClient
    from debugging import get_debugger

    # Import conditionnel des modules avancés
    try:
        from _engine import ResponseValidator
        VALIDATION_AVAILABLE = True
    except ImportError:
        VALIDATION_AVAILABLE = False

    try:
        from quality_monitor import QualityMonitor
        MONITORING_AVAILABLE = True
    except ImportError:
        MONITORING_AVAILABLE = False

    MODULES_LOADED = True
    print("✅ Modules Medical RAG importés avec succès")

except ImportError as e:
    MODULES_LOADED = False
    print(f"❌ Échec de l'import des modules Medical RAG: {e}")
    print("Assurez-vous que tous les fichiers Python requis sont dans le même répertoire")

def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_directories():
    """S'assure que les répertoires requis existent"""
    directories = [
        'uploaded_docs/permanent',
        'uploaded_docs/temporary',
        'system_files/logs',
        'system_files/cache'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def initialize_rag_system(profile='balanced'):
    """Initialise le système RAG avec gestion d'erreurs"""
    global rag_system, debugger

    if not MODULES_LOADED:
        raise Exception("Modules requis non chargés")

    try:
        # Initialiser le debugger
        debugger = get_debugger()
        debugger.log_info("Initialisation du serveur web RAG", "RAG web server initialization")

        # S'assurer que les répertoires existent
        ensure_directories()

        # Initialiser le système RAG
        rag_system = RAGSystem(
            profile=profile,
            enable_validation=VALIDATION_AVAILABLE,
            enable_monitoring=MONITORING_AVAILABLE
        )

        # Charger les documents existants
        rag_system.load_all_documents()

        debugger.log_info("Système RAG initialisé avec succès", "RAG system initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Échec de l'initialisation du système RAG: {e}")
        traceback.print_exc()
        return False

# Initialiser au démarrage
if MODULES_LOADED:
    initialization_success = initialize_rag_system()
else:
    initialization_success = False

# ROUTES POUR LES FICHIERS STATIQUES - ORDRE IMPORTANT
@app.route('/')
def serve_index():
    """Servir l'interface web principale"""
    try:
        return send_file('index.html', mimetype='text/html')
    except FileNotFoundError:
        return jsonify({'error': 'Interface web non trouvée'}), 404

@app.route('/app.js')
def serve_app_js():
    """Servir le fichier JavaScript principal"""
    try:
        return send_file('app.js', mimetype='application/javascript')
    except FileNotFoundError:
        return jsonify({'error': 'app.js non trouvé'}), 404

@app.route('/style.css')
def serve_style_css():
    """Servir le fichier CSS principal"""
    try:
        return send_file('style.css', mimetype='text/css')
    except FileNotFoundError:
        return jsonify({'error': 'style.css non trouvé'}), 404

# Route générale pour autres fichiers statiques (images, etc.)
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    """Servir les fichiers statiques depuis le répertoire static"""
    try:
        return send_from_directory('static', filename)
    except FileNotFoundError:
        return jsonify({'error': f'Fichier statique {filename} non trouvé'}), 404

# Servir les PDFs pour la visualisation
@app.route('/api/documents/pdf/<path:document_name>')
def serve_pdf(document_name):
    """Servir un document PDF pour la visualisation"""
    # Sécuriser le nom du document
    document_name = secure_filename(document_name)

    # Chercher dans les deux dossiers
    for folder in ['uploaded_docs/permanent', 'uploaded_docs/temporary']:
        path = os.path.join(folder, document_name)
        if os.path.exists(path) and path.endswith('.pdf'):
            try:
                return send_file(path, mimetype='application/pdf')
            except Exception as e:
                return jsonify({'error': f'Erreur lors de la lecture du PDF: {str(e)}'}), 500

    return jsonify({'error': 'PDF non trouvé'}), 404

# ENDPOINTS API
@app.route('/api', methods=['GET'])
def api_documentation():
    """Retourner la documentation des endpoints API disponibles"""
    endpoints = []
    for rule in app.url_map.iter_rules():
        if 'static' not in rule.rule and 'api' in rule.rule:
            methods = list(rule.methods - {'HEAD', 'OPTIONS'})
            endpoints.append({
                'endpoint': rule.rule,
                'methods': methods
            })

    return jsonify({
        'api_version': '1.0',
        'endpoints': endpoints,
        'rag_system_ready': rag_system is not None,
        'modules_loaded': MODULES_LOADED,
        'validation_available': VALIDATION_AVAILABLE,
        'monitoring_available': MONITORING_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé du système"""
    global rag_system

    health_status = {
        'status': 'ok' if rag_system is not None else 'error',
        'timestamp': datetime.now().isoformat(),
        'modules_loaded': MODULES_LOADED,
        'rag_system_initialized': rag_system is not None,
        'validation_available': VALIDATION_AVAILABLE,
        'monitoring_available': MONITORING_AVAILABLE,
        'uptime_seconds': time.time() - system_start_time
    }

    if rag_system:
        try:
            system_status = rag_system.get_system_status()
            health_status.update({
                'system_status': system_status,
                'backend_ready': True
            })
        except Exception as e:
            health_status.update({
                'backend_ready': False,
                'error': str(e)
            })
    else:
        health_status['backend_ready'] = False

    return jsonify(health_status)

@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    """Obtenir le statut détaillé du système"""
    if not rag_system:
        return jsonify({'error': 'Système RAG non initialisé'}), 500

    try:
        status = rag_system.get_system_status()

        # Ajouter des informations spécifiques au web
        status['web_interface'] = {
            'backend_uptime': time.time() - system_start_time,
            'modules_loaded': MODULES_LOADED,
            'validation_available': VALIDATION_AVAILABLE,
            'monitoring_available': MONITORING_AVAILABLE
        }

        return jsonify(status)

    except Exception as e:
        if debugger:
            debugger.log_error("Erreur lors de la récupération du statut", "Error getting system status", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """Obtenir la liste des documents chargés"""
    if not rag_system:
        return jsonify({'error': 'Système RAG non initialisé'}), 500

    try:
        # Obtenir les documents par type
        permanent_sources = rag_system.memory.list_sources_by_type("permanent")
        temporary_sources = rag_system.memory.list_sources_by_type("temporary")
        all_sources = rag_system.memory.list_sources_in_db()

        # Obtenir les détails des documents si disponibles
        documents = {
            'permanent': [],
            'temporary': [],
            'all': all_sources,
            'total_count': len(all_sources)
        }

        # Ajouter les détails des fichiers pour les documents permanents
        for source in permanent_sources:
            file_path = os.path.join('uploaded_docs/permanent', source)
            file_info = {
                'name': source,
                'type': 'permanent',
                'exists': os.path.exists(file_path)
            }

            if file_info['exists']:
                stat = os.stat(file_path)
                file_info.update({
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

            documents['permanent'].append(file_info)

        # Ajouter les détails des fichiers pour les documents temporaires
        for source in temporary_sources:
            file_path = os.path.join('uploaded_docs/temporary', source)
            file_info = {
                'name': source,
                'type': 'temporary',
                'exists': os.path.exists(file_path)
            }

            if file_info['exists']:
                stat = os.stat(file_path)
                file_info.update({
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

            documents['temporary'].append(file_info)

        return jsonify(documents)

    except Exception as e:
        if debugger:
            debugger.log_error("Erreur lors de la liste des documents", "Error listing documents", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/load', methods=['POST'])
def load_documents():
    """Charger les documents depuis un dossier spécifié"""
    if not rag_system:
        return jsonify({'error': 'Système RAG non initialisé'}), 500

    data = request.get_json()
    folder_type = data.get('folder', 'all')  # 'permanent', 'temporary', ou 'all'

    try:
        if folder_type == 'all':
            permanent_loaded, temporary_loaded = rag_system.load_all_documents()
            result = {
                'success': True,
                'permanent_loaded': permanent_loaded,
                'temporary_loaded': temporary_loaded,
                'message': 'Tous les documents chargés'
            }

        elif folder_type == 'permanent':
            success = rag_system.load_documents('uploaded_docs/permanent')
            result = {
                'success': success,
                'message': 'Documents permanents chargés' if success else 'Aucun document permanent trouvé'
            }

        elif folder_type == 'temporary':
            success = rag_system.load_documents('uploaded_docs/temporary')
            result = {
                'success': success,
                'message': 'Documents temporaires chargés' if success else 'Aucun document temporaire trouvé'
            }

        else:
            return jsonify({'error': 'Type de dossier invalide'}), 400

        # Obtenir la liste mise à jour des documents
        updated_documents = rag_system.memory.list_sources_in_db()
        result['total_documents'] = len(updated_documents)

        if debugger:
            debugger.log_info(f"Documents chargés: {folder_type}", f"Documents loaded: {folder_type}")

        return jsonify(result)

    except Exception as e:
        if debugger:
            debugger.log_error("Erreur lors du chargement des documents", "Error loading documents", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Gérer l'upload de fichiers"""
    if not rag_system:
        return jsonify({'error': 'Système RAG non initialisé'}), 500

    if 'files' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400

    files = request.files.getlist('files')
    doc_type = request.form.get('type', 'temporary')  # permanent ou temporary

    if doc_type not in ['permanent', 'temporary']:
        return jsonify({'error': 'Type de document invalide'}), 400

    uploaded_files = []
    errors = []

    try:
        target_dir = os.path.join('uploaded_docs', doc_type)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                errors.append(f"Type de fichier non autorisé: {file.filename}")
                continue

            filename = secure_filename(file.filename)
            file_path = os.path.join(target_dir, filename)

            # Sauvegarder le fichier
            file.save(file_path)

            uploaded_files.append({
                'name': filename,
                'path': file_path,
                'type': doc_type,
                'size_bytes': os.path.getsize(file_path),
                'uploaded_at': datetime.now().isoformat()
            })

        # Recharger les documents pour inclure les nouveaux uploads
        if doc_type == 'permanent':
            rag_system.load_documents('uploaded_docs/permanent')
        else:
            rag_system.load_documents('uploaded_docs/temporary')

        result = {
            'success': True,
            'uploaded_files': uploaded_files,
            'errors': errors,
            'message': f"{len(uploaded_files)} fichiers uploadés avec succès"
        }

        if debugger:
            debugger.log_info(f"{len(uploaded_files)} fichiers uploadés", f"{len(uploaded_files)} files uploaded")

        return jsonify(result)

    except Exception as e:
        if debugger:
            debugger.log_error("Erreur lors de l'upload", "Error during upload", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def process_question():
    """Traiter une question via le système RAG"""
    if not rag_system:
        return jsonify({'error': 'Système RAG non initialisé'}), 500

    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'La question est requise'}), 400

    # Analyser les paramètres optionnels
    filters = data.get('filters', {})
    debug_mode = data.get('debug', False)
    bypass_cache = data.get('bypass_cache', False)

    # Analyser la syntaxe de requête avancée (doc:, type:, debug:)
    where_filter = None
    if question.startswith('doc:'):
        parts = question.split(maxsplit=1)
        if len(parts) == 2:
            doc_name = parts[0][4:]  # Supprimer "doc:"
            question = parts[1]
            where_filter = {"source": doc_name}

    elif question.startswith('type:'):
        parts = question.split(maxsplit=1)
        if len(parts) == 2:
            doc_type = parts[0][5:]  # Supprimer "type:"
            question = parts[1]
            where_filter = {"doc_type": doc_type}

    elif question.startswith('debug:'):
        question = question[6:].strip()
        debug_mode = True

    # Appliquer des filtres supplémentaires
    if filters.get('document'):
        where_filter = {"source": filters['document']}
    elif filters.get('type'):
        where_filter = {"doc_type": filters['type']}

    try:
        if debugger:
            debugger.log_info(f"Traitement de la question: {question[:50]}...", f"Processing question: {question[:50]}...")

        # Traiter la question via le système RAG
        result = rag_system.ask_question(
            question=question,
            where=where_filter,
            debug=debug_mode,
            bypass_cache=bypass_cache
        )

        # Formater la réponse pour l'interface web
        web_result = {
            'question': result['question'],
            'answer': result['answer'],
            'sources': [],
            'performance': result.get('performance', {}),
            'validation': result.get('validation', {}),
            'metadata': result.get('metadata', {}),
            'debug_info': {}
        }

        # Traiter les sources
        if 'relevant_passages' in result:
            for idx, (doc, score) in result['relevant_passages']:
                source_info = {
                    'index': idx,
                    'score': round(score, 3),
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }

                if hasattr(doc, 'metadata'):
                    source_info.update({
                        'document': doc.metadata.get('source', 'Unknown'),
                        'page': doc.metadata.get('page', 'N/A'),
                        'type': doc.metadata.get('doc_type', 'unknown'),
                        'headings': doc.metadata.get('headings', [])
                    })

                    # Parser les headings si c'est une chaîne JSON
                    if isinstance(source_info['headings'], str):
                        try:
                            source_info['headings'] = json.loads(source_info['headings'])
                        except:
                            source_info['headings'] = []

                web_result['sources'].append(source_info)

        # Ajouter des informations de debug si demandé
        if debug_mode:
            web_result['debug_info'] = {
                'filters_applied': where_filter,
                'bypass_cache': bypass_cache,
                'total_passages_found': len(result.get('relevant_passages', [])),
                'settings_profile': rag_system.settings.profile
            }

        # Ajouter des suggestions d'amélioration si disponibles
        if 'improvement_suggestions' in result:
            web_result['improvement_suggestions'] = result['improvement_suggestions']

        if debugger:
            debugger.log_info("Question traitée avec succès", "Question processed successfully")

        return jsonify(web_result)

    except Exception as e:
        if debugger:
            debugger.log_error("Erreur lors du traitement de la question", "Error processing question", e)

        return jsonify({
            'error': str(e),
            'question': question,
            'traceback': traceback.format_exc() if debug_mode else None
        }), 500

# Gestionnaires d'erreurs
@app.errorhandler(404)
def not_found(error):
    """Gérer les erreurs 404 avec plus de détails"""
    requested_path = request.path

    # Ne pas renvoyer de JSON pour les requêtes de fichiers statiques
    if any(ext in requested_path.lower() for ext in ['.js', '.css', '.png', '.jpg', '.ico']):
        return f"File not found: {requested_path}", 404

    available_endpoints = [
        rule.rule for rule in app.url_map.iter_rules()
        if 'static' not in rule.rule and 'api' in rule.rule
    ]

    return jsonify({
        'error': f'Endpoint {requested_path} non trouvé',
        'available_endpoints': available_endpoints,
        'suggestion': 'Consultez /api pour tous les endpoints disponibles'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Gérer les erreurs 500"""
    return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.errorhandler(413)
def too_large(error):
    """Gérer les erreurs de fichier trop volumineux"""
    return jsonify({'error': 'Fichier trop volumineux (max 100MB)'}), 413

# Point d'entrée principal
if __name__ == '__main__':
    print("🚀 Démarrage du serveur web Medical RAG System")
    print(f"📁 Répertoire de travail: {os.getcwd()}")
    print(f"🔧 Modules chargés: {MODULES_LOADED}")
    print(f"✅ Système RAG prêt: {initialization_success}")

    if MODULES_LOADED and initialization_success:
        print("🌐 Serveur démarré sur http://localhost:8000")
        print("📱 Ouvrez http://localhost:8000 dans votre navigateur pour accéder à l'interface")
    else:
        print("⚠️ Serveur démarré avec des fonctionnalités limitées")
        print("💡 Assurez-vous que tous les fichiers Python sont dans le même répertoire et que les dépendances sont installées")

    # Démarrer le serveur de développement Flask
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,  # Désactivé pour éviter les conflits
        threaded=True
    )