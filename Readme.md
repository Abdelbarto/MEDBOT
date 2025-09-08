# MedRAG – Système RAG Médical Haute Performance

MedRAG est un système RAG (Retrieval-Augmented Generation) optimisé pour les applications médicales. Il combine un index vectoriel, un LLM local (Ollama), un module de validation médicale et un système de monitoring de qualité, le tout enrichi d’un mécanisme de debugging avancé.

***

## Table des matières

- [Fonctionnalités](#fonctionnalit%C3%A9s)
- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Debugging \& Logging](#debugging--logging)
- [Cache \& Base de données](#cache--base-de-donn%C3%A9es)
- [Contribuer](#contribuer)
- [Licence](#licence)

***

## Fonctionnalités

- **Récupération de passages** par similarité vectorielle (Chroma + OllamaEmbeddings)
- **Génération** de réponses structurées via un LLM local Ollama
- **Validation médicale** post-génération (cohérence, citations, alignement factuel)
- **Monitoring de qualité** avec métriques (temps, taux de succès, score de confiance)
- **Debugging avancé** : logs bilingues, timing, tracing, export JSON
- **Cache intelligent** : SQLite + en mémoire, TTL configurable

***

## Architecture du projet

```
.
├── main.py                # Point d’entrée de l’application
├── RAGSystem.py           # Pipeline RAG complet
├── memory_builder.py      # Construction et gestion du store vectoriel + cache
├── llm_client.py          # Client LLM Ollama avec retry et métriques
├── engine.py              # Validation des réponses (ResponseValidator)
├── quality_monitor.py     # Monitoring de qualité (QualityMonitor)
├── debugging.py           # Système de debugging et timing
├── grader.py              # Évaluation de pertinence (RetrievalGrader)
├── model_settings.py      # Profils (performance, balanced, accuracy)
├── system_files/
│   ├── cache/
│   │   ├── query_cache.db # Cache SQL des requêtes
│   └── logs/
│       ├── medical_rag_debug.log          # Logs de debugging
│       └── medical_rag_quality.log        # Logs de monitoring
├── uploaded_docs/         # Documents PDF/TXT à indexer
│   ├── permanent/
│   └── temporary/
└── README.md              # Documentation du projet
```


***

## Installation

1. Clonez le dépôt :

```bash
git clone https://github.com/votre-org/MedRAG.git
cd MedRAG
```

2. Créez un environnement virtuel et installez les dépendances :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Assurez-vous d’avoir Ollama installé et un modèle local configuré.

***

## Configuration

- **Profils** définis dans `model_settings.py` :
    - `performance` : vitesse maximale, validation minimale
    - `balanced` : compromis vitesse/précision
    - `accuracy` : précision maximale, validation complète
- Paramètres de cache et TTL dans `MemoryBuilder.__init__`
- Activer/désactiver validation et monitoring via `RAGSystem(profile=..., enable_validation=..., enable_monitoring=...)`

***

## Utilisation

1. Lancez l’application :

```bash
source .venv/bin/activate
python app.py
```

2. Options :
    - `help` : afficher l’aide interactive
    - `settings` : menu de configuration (retrieval, génération, chunking…)
    - `status` : afficher le statut système et métriques
    - `quit` : quitter
3. Syntaxes avancées :
    - `doc:nom.pdf question` : recherche dans un fichier spécifique
    - `type:permanent question` : uniquement dans `uploaded_docs/permanent`
    - `debug: question` : active le mode debug

***

## Debugging \& Logging

- Tous les logs sont écrits dans `system_files/logs/medical_rag_debug.log`.
- Monitoring (performance, qualité) : `system_files/logs/medical_rag_quality.log`.
- Configuration du logger dans `debugging.py` utilise uniquement `FileHandler`.
- Export des données de debug au format JSON via `debugger.export_debug_data()`.

***

## Cache \& Base de données

- **Requête cache** :
    - Stockée dans `system_files/cache/query_cache.db`.
    - TTL configurable (`cache_ttl_hours`).
- **Monitoring DB** (QualityMonitor) :
    - `quality_metrics.db` dans le répertoire courant.

***

## Contribuer

1. Fork \& clone le projet
2. Créez une branche feature :

```bash
git checkout -b feature/ma-amelioration
```

3. Développez et testez
4. Ouvrez une pull request vers la branche `main`

***

## Licence

MIT License. Voir le fichier `LICENSE` pour plus de détails.

<div style="text-align: center">⁂</div>

[^1]: engine.py

[^2]: grader.py

[^3]: llm_client.py

[^4]: main.py

[^5]: memory_builder.py

[^6]: model_settings.py

[^7]: quality_monitor.py

[^8]: RAGSystem.py

