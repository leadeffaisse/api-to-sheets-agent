import requests
import os
from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
import re
from datetime import datetime

# Chargement des variables d'environnement
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import gspread
from google.oauth2.service_account import Credentials

# =============================================================================
# CONFIGURATION DEPUIS .ENV AVEC VALEURS PAR DÉFAUT
# =============================================================================

# API Keys (SENSIBLE - depuis .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Configuration OpenAI (CONFIGURABLE - depuis .env avec défauts)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

# Google Configuration (SENSIBLE/CONFIGURABLE - depuis .env avec défauts)
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "./google-credentials.json")
GOOGLE_PERSONAL_EMAIL = os.getenv("GOOGLE_PERSONAL_EMAIL")

# API Configuration (CONFIGURABLE - depuis .env avec défauts)
DEFAULT_API_URL = os.getenv("DEFAULT_API_URL", "https://jsonplaceholder.typicode.com/posts")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Limites métier (CONFIGURABLE - depuis .env avec défauts)
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "10"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "100"))
MIN_LIMIT = int(os.getenv("MIN_LIMIT", "1"))

# Google Sheets (CONFIGURABLE - depuis .env avec défauts)
SHEETS_FOLDER_NAME = os.getenv("SHEETS_FOLDER_NAME", "API_Data_Exports")
SHEETS_SHARE_PUBLICLY = os.getenv("SHEETS_SHARE_PUBLICLY", "false").lower() == "true"
SHEETS_DEFAULT_TITLE_PREFIX = os.getenv("SHEETS_DEFAULT_TITLE_PREFIX", "API_Data")

# Debug et logging (CONFIGURABLE - depuis .env avec défauts)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# LangGraph Studio Configuration (CONFIGURABLE - depuis .env avec défauts)
BG_JOB_ISOLATED_LOOPS = os.getenv("BG_JOB_ISOLATED_LOOPS", "true").lower() == "true"
LANGGRAPH_STUDIO_DEBUG = os.getenv("LANGGRAPH_STUDIO_DEBUG", "true").lower() == "true"
LANGGRAPH_STUDIO_ASYNC = os.getenv("LANGGRAPH_STUDIO_ASYNC", "true").lower() == "true"

# =============================================================================
# CONFIGURATION LANGSMITH
# =============================================================================

try:
    from langsmith import Client, trace, traceable
    from langchain_core.tracers import LangChainTracer
    
    # Configuration LangSmith
    LANGSMITH_CONFIG = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "api-to-sheets-agent"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGCHAIN_API_KEY": LANGSMITH_API_KEY or ""
    }
    
    # Appliquer la configuration LangSmith
    for key, value in LANGSMITH_CONFIG.items():
        if value:
            os.environ[key] = value
    
    # Client LangSmith
    langsmith_client = Client() if LANGSMITH_API_KEY else None
    langsmith_available = bool(LANGSMITH_API_KEY)
    
    if langsmith_available:
        print("✅ LangSmith configuré - Tracking tokens automatique activé")
    else:
        print("⚠️ LangSmith non configuré")
    
except ImportError:
    print("⚠️ LangSmith non installé - tracking tokens désactivé")
    langsmith_client = None
    langsmith_available = False
    trace = None
    traceable = None

# =============================================================================
# FONCTIONS UTILITAIRES POUR TRACING MODERNE
# =============================================================================

def create_trace_context(name: str, tags: list = None, metadata: dict = None):
    """Crée un contexte de trace compatible avec l'API moderne (sans timeout)"""
    if not langsmith_available or not trace:
        return None
    
    try:
        return trace(
            name=name,
            tags=tags or [],
            metadata=metadata or {}
        )
    except Exception as e:
        log_debug(f"Erreur création trace (ignorée): {e}")
        return None

def safe_trace_update(trace_context, **kwargs):
    """Met à jour une trace de manière sécurisée"""
    if trace_context:
        try:
            if hasattr(trace_context, 'update'):
                trace_context.update(**kwargs)
            elif hasattr(trace_context, 'add_metadata'):
                trace_context.add_metadata(kwargs)
        except Exception as e:
            log_debug(f"Erreur mise à jour trace (ignorée): {e}")

class DummyContext:
    """Context manager dummy pour les cas où le tracing n'est pas disponible"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, **kwargs):
        pass

# =============================================================================
# CONFIGURATION DU LLM AVEC NOUVEAU TRACING
# =============================================================================

callbacks = []
if langsmith_available:
    try:
        # ✅ LE LANGCHAIN TRACER CAPTURE AUTOMATIQUEMENT :
        # - Tous les tokens (prompt + completion)
        # - Les coûts calculés automatiquement
        # - Les métriques de performance
        # - Les erreurs et timeouts
        callbacks.append(LangChainTracer(project_name=LANGSMITH_CONFIG["LANGCHAIN_PROJECT"]))
        print(f"✅ LangChain Tracer configuré pour le projet: {LANGSMITH_CONFIG['LANGCHAIN_PROJECT']}")
        print("💰 Tracking automatique des tokens et coûts activé")
    except Exception as tracer_error:
        print(f"⚠️ Erreur configuration LangChain Tracer: {tracer_error}")

if OPENAI_API_KEY:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=OPENAI_TEMPERATURE,
        timeout=API_TIMEOUT,
        max_retries=MAX_RETRIES,
        callbacks=callbacks  # ← C'est tout ce qu'il faut !
    )
else:
    llm = None

# =============================================================================
# FONCTIONS UTILITAIRES POUR RÉCUPÉRER LES STATS DEPUIS LANGSMITH
# =============================================================================

def get_langsmith_token_stats(project_name: str = None, hours: int = 24) -> dict:
    """Récupère les statistiques de tokens depuis LangSmith"""
    if not langsmith_available:
        return {"error": "LangSmith non disponible"}
    
    try:
        project = project_name or LANGSMITH_CONFIG["LANGCHAIN_PROJECT"]
        
        # Récupérer les runs récents
        runs = langsmith_client.list_runs(
            project_name=project,
            limit=100,
            # Filtrer les dernières heures
        )
        
        total_tokens = 0
        total_cost = 0.0
        run_count = 0
        
        for run in runs:
            if hasattr(run, 'extra') and run.extra:
                # LangSmith stocke les métriques dans run.extra
                usage = run.extra.get('metadata', {}).get('ls_usage', {})
                if usage:
                    total_tokens += usage.get('total_tokens', 0)
                    total_cost += usage.get('total_cost', 0.0)
                    run_count += 1
        
        return {
            "success": True,
            "total_runs": run_count,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "average_tokens_per_run": total_tokens / max(run_count, 1),
            "project": project
        }
        
    except Exception as e:
        return {"error": f"Erreur récupération stats: {str(e)}"}

def log_langsmith_token_summary():
    """Affiche un résumé des tokens depuis LangSmith"""
    stats = get_langsmith_token_stats()
    
    if "error" in stats:
        log_debug(f"⚠️ {stats['error']}")
        return
    
    print(f"\n💰 === STATISTIQUES LANGSMITH ===")
    print(f"📊 Runs analysés: {stats['total_runs']}")
    print(f"🔤 Tokens totaux: {stats['total_tokens']:,}")
    print(f"💵 Coût total: ${stats['total_cost_usd']:.4f}")
    print(f"📈 Moyenne par run: {stats['average_tokens_per_run']:.1f} tokens")
    print(f"🎯 Projet: {stats['project']}")
    print(f"🔗 Voir détails: https://smith.langchain.com/projects/{stats['project']}")
    print(f"=================================\n")

# =============================================================================
# CONFIGURATION TECHNIQUE (CONSTANTES - RESTE DANS LE CODE)
# =============================================================================

# Champs API disponibles (LOGIQUE MÉTIER - dans le code)
VALID_API_FIELDS = ["userId", "id", "title", "body"]

# Mots-clés pour le parsing (LOGIQUE MÉTIER - dans le code)
FIELD_KEYWORDS = {
    "title": ["title", "titre"],
    "id": ["id", "identifiant"],
    "userId": ["userid", "user", "utilisateur"],
    "body": ["body", "contenu", "texte"]
}

RESTRICTION_KEYWORDS = ["avec", "seulement", "uniquement", "juste"]

# Google Sheets Scopes (TECHNIQUE - dans le code)
GOOGLE_SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Patterns regex (TECHNIQUE - dans le code)
JSON_EXTRACTION_PATTERN = r'\{.*\}'
NUMBER_EXTRACTION_PATTERN = r'\b(\d+)\b'

# =============================================================================
# VALIDATION DES VARIABLES CRITIQUES
# =============================================================================

def validate_environment():
    """Valide que toutes les variables critiques sont présentes"""
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY manquante dans .env")
        return False
    
    if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        print(f"⚠️ Fichier de credentials Google introuvable: {GOOGLE_CREDENTIALS_PATH}")
        return False
    
    return True

# Valider au chargement du module
env_valid = validate_environment()

# =============================================================================
# ÉTAT TYPÉ POUR LANGGRAPH
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    api_url: str
    user_query: str
    extracted_params: Optional[Dict[str, Any]]
    api_data: Optional[List[Dict]]
    processed_data: Optional[List[Dict]]
    sheets_url: str
    error: str

# =============================================================================
# CONFIGURATION GOOGLE SHEETS
# =============================================================================

def setup_google_sheets():
    """Configuration de l'accès Google Sheets"""
    try:
        if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
            print(f"⚠️ Fichier credentials Google introuvable: {GOOGLE_CREDENTIALS_PATH}")
            return None
            
        creds = Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_PATH, scopes=GOOGLE_SCOPES
        )
        return gspread.authorize(creds)
    except Exception as e:
        print(f"⚠️ Erreur configuration Google Sheets: {e}")
        return None

gc = setup_google_sheets()

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def log_debug(message: str):
    """Log de debug si activé"""
    if DEBUG:
        print(f"🔍 DEBUG: {message}")

def ensure_state_keys(state: AgentState) -> AgentState:
    """S'assurer que toutes les clés nécessaires sont présentes dans l'état"""
    default_state = {
        "messages": [],
        "api_url": DEFAULT_API_URL,
        "user_query": "",
        "extracted_params": None,
        "api_data": None,
        "processed_data": None,
        "sheets_url": "",
        "error": ""
    }
    
    for key, default_value in default_state.items():
        if key not in state:
            state[key] = default_value
    
    return state

# =============================================================================
# FONCTIONS PRINCIPALES (DÉFINIES AVANT build_graph)
# =============================================================================

def validate_extracted_params(params: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """Valide et corrige AGRESSIVEMENT les paramètres extraits"""
    
    # ✅ CORRECTION : Pas de timeout
    trace_context = create_trace_context(
        name="validate_extracted_params",
        tags=["validation", "parameters"],
        metadata={"step": "1.1", "component": "parameter_validator"}
    )
    
    try:
        with trace_context or DummyContext():
            user_query_lower = user_query.lower()
            log_debug(f"Validation pour: '{user_query_lower}'")
            
            safe_trace_update(trace_context, inputs={"raw_params": params, "user_query": user_query})
            
            # Vérifier que params est un dictionnaire valide
            if not isinstance(params, dict):
                log_debug(f"Params invalide (type: {type(params)}), création d'un nouveau dict")
                params = {}
            
            # 1. VALIDATION DU LIMIT
            try:
                numbers = re.findall(NUMBER_EXTRACTION_PATTERN, user_query)
                if numbers:
                    limit = int(numbers[0])
                    limit = max(MIN_LIMIT, min(limit, MAX_LIMIT))
                    params["limit"] = limit
                    log_debug(f"Limite corrigée: {params['limit']}")
                elif "limit" not in params or not isinstance(params.get("limit"), int) or params.get("limit", 0) <= 0:
                    params["limit"] = DEFAULT_LIMIT
                    log_debug(f"Limite par défaut: {params['limit']}")
            except Exception as limit_error:
                log_debug(f"Erreur validation limit: {limit_error}")
                params["limit"] = DEFAULT_LIMIT
            
            # 2. VALIDATION DES FIELDS
            try:
                mentioned_fields = []
                for field, keywords in FIELD_KEYWORDS.items():
                    if any(f" {keyword} " in f" {user_query_lower} " or 
                           user_query_lower.startswith(keyword + " ") or 
                           user_query_lower.endswith(" " + keyword) for keyword in keywords):
                        mentioned_fields.append(field)
                
                has_restriction_keywords = any(word in user_query_lower for word in RESTRICTION_KEYWORDS)
                
                if mentioned_fields and has_restriction_keywords:
                    params["fields"] = mentioned_fields
                elif not mentioned_fields:
                    params["fields"] = VALID_API_FIELDS[:]
                else:
                    if "fields" not in params or not isinstance(params.get("fields"), list):
                        params["fields"] = VALID_API_FIELDS[:]
                    else:
                        params["fields"] = [field for field in params["fields"] if field in VALID_API_FIELDS]
                        if not params["fields"]:
                            params["fields"] = VALID_API_FIELDS[:]
            except Exception as fields_error:
                log_debug(f"Erreur validation fields: {fields_error}")
                params["fields"] = VALID_API_FIELDS[:]
            
            # 3. VALIDATION DES FILTERS
            try:
                if "filters" not in params or not isinstance(params.get("filters"), dict):
                    params["filters"] = {}
            except Exception as filters_error:
                log_debug(f"Erreur validation filters: {filters_error}")
                params["filters"] = {}
            
            # 4. VALIDATION DE LA DESCRIPTION
            try:
                if "description" not in params or not isinstance(params.get("description"), str):
                    params["description"] = f"Récupération de {params['limit']} posts avec les champs {', '.join(params['fields'])}"
            except Exception as desc_error:
                log_debug(f"Erreur validation description: {desc_error}")
                params["description"] = f"Récupération de données"
            
            safe_trace_update(trace_context, outputs={"validated_params": params})
            
            log_debug(f"Validation terminée avec succès: {params}")
            return params
    
    except Exception as e:
        log_debug(f"Erreur dans validate_extracted_params: {type(e).__name__}: {str(e)}")
        # En cas d'erreur, retourner des paramètres par défaut valides
        fallback_params = {
            "limit": DEFAULT_LIMIT,
            "fields": VALID_API_FIELDS[:],
            "filters": {},
            "description": f"Paramètres par défaut suite à une erreur de validation"
        }
        log_debug(f"Retour de paramètres fallback: {fallback_params}")
        return fallback_params

def parse_user_query(state: AgentState) -> AgentState:
    """Parse la requête utilisateur pour extraire les paramètres"""
    
    log_debug(f"=== DÉBUT PARSE_USER_QUERY ===")
    
    # ✅ CORRECTION : Pas de timeout
    trace_context = create_trace_context(
        name="parse_user_query",
        tags=["parsing", "user_input"],
        metadata={"step": "1", "component": "query_parser"}
    )
    
    try:
        with trace_context or DummyContext():
            log_debug(f"VERSION MISE À JOUR CHARGÉE - TIMESTAMP: {datetime.now()}")
            
            # Récupérer la dernière requête utilisateur
            user_query = ""
            messages = state.get("messages", [])
            
            for message in reversed(messages):
                if isinstance(message, dict):
                    if message.get('type') == 'human' and message.get('content'):
                        user_query = message['content']
                        break
                elif isinstance(message, HumanMessage):
                    user_query = message.content
                    break
            
            safe_trace_update(trace_context, 
                inputs={"user_query": user_query, "messages_count": len(messages)}
            )
            
            log_debug(f"Requête à analyser: '{user_query}'")
            
            if not llm:
                state["error"] = "LLM non configuré - vérifiez OPENAI_API_KEY"
                log_debug(f"=== FIN PARSE_USER_QUERY (erreur LLM) ===")
                return state
            
            if not user_query.strip():
                state["error"] = "Requête utilisateur vide"
                log_debug(f"=== FIN PARSE_USER_QUERY (requête vide) ===")
                return state
            
            prompt = ChatPromptTemplate.from_template(
                "Analyse la requête utilisateur et génère un JSON structuré pour requête API.\n"
                "Requête: {user_query}\n"
                "Réponds uniquement avec le JSON contenant les clés: limit, fields, filters, description."
            )

            parser = JsonOutputParser()

            chain = prompt | llm | parser

            log_debug("⚡ Appel du LLM pour parsing de la requête utilisateur")
            params = chain.invoke({"user_query": user_query})
            
            # Validation et nettoyage des paramètres
            log_debug("Début validation des paramètres")
            validated_params = validate_extracted_params(params, user_query)
            log_debug("Fin validation des paramètres")
            
            state["extracted_params"] = validated_params
            state["user_query"] = user_query
            
            # Ne pas définir d'erreur si tout va bien
            if "error" in state:
                del state["error"]
            
            safe_trace_update(trace_context,
                outputs={
                    "extracted_params": validated_params,
                    "parsing_success": True
                }
            )
            
            log_debug(f"Paramètres finaux: {validated_params}")
            log_debug(f"=== FIN PARSE_USER_QUERY (succès) ===")
                
    except Exception as e:
        error_msg = f"Erreur lors du parsing: {str(e)}"
        log_debug(f"Exception dans parse_user_query: {type(e).__name__}: {str(e)}")
        
        # Essayer de créer des paramètres fallback même en cas d'erreur
        try:
            log_debug("Tentative de création de paramètres fallback d'urgence")
            params = create_fallback_params(user_query if 'user_query' in locals() else "récupérer des posts")
            validated_params = validate_extracted_params(params, user_query if 'user_query' in locals() else "")
            state["extracted_params"] = validated_params
            state["user_query"] = user_query if 'user_query' in locals() else ""
            log_debug(f"Paramètres fallback d'urgence créés: {validated_params}")
            # Ne pas définir d'erreur si on a pu créer des paramètres
            if "error" in state:
                del state["error"]
        except Exception as fallback_error:
            log_debug(f"Erreur création fallback d'urgence: {fallback_error}")
            state["error"] = error_msg
        
        safe_trace_update(trace_context,
            outputs={"error": error_msg, "parsing_success": False}
        )
        
        log_debug(f"=== FIN PARSE_USER_QUERY (erreur) ===")
    
    return state

def create_fallback_params(user_query: str) -> Dict[str, Any]:
    """Crée des paramètres par défaut basés sur une analyse simple de la requête"""
    
    log_debug(f"Création de paramètres fallback pour: '{user_query}'")
    
    try:
        # Analyse simple pour extraire le nombre
        numbers = re.findall(r'\b(\d+)\b', user_query)
        limit = int(numbers[0]) if numbers else DEFAULT_LIMIT
        limit = max(MIN_LIMIT, min(limit, MAX_LIMIT))
        
        # Analyse simple pour les champs
        user_query_lower = user_query.lower()
        mentioned_fields = []
        
        for field, keywords in FIELD_KEYWORDS.items():
            if any(keyword in user_query_lower for keyword in keywords):
                mentioned_fields.append(field)
        
        # Vérifier les mots de restriction
        has_restriction = any(word in user_query_lower for word in RESTRICTION_KEYWORDS)
        
        if mentioned_fields and has_restriction:
            fields = mentioned_fields[:]  # Copie de la liste
        else:
            fields = VALID_API_FIELDS[:]  # Copie de la liste
        
        params = {
            "limit": limit,
            "fields": fields,
            "filters": {},
            "description": f"Récupération de {limit} posts avec les champs {', '.join(fields)} (fallback)"
        }
        
        log_debug(f"Paramètres fallback créés: {params}")
        return params
        
    except Exception as e:
        log_debug(f"Erreur dans create_fallback_params: {e}")
        # Paramètres d'urgence
        return {
            "limit": DEFAULT_LIMIT,
            "fields": VALID_API_FIELDS[:],
            "filters": {},
            "description": "Paramètres d'urgence"
        }

def fetch_api_data(state: AgentState) -> AgentState:
    """Récupère les données depuis l'API"""
    
    trace_context = None
    if langsmith_client:
        try:
            trace_context = langsmith_client.trace(
                name="fetch_api_data",
                tags=["api", "data_fetching"],
                metadata={"step": "2", "component": "api_client"}
            ).__enter__()
        except:
            pass
    
    try:
        state = ensure_state_keys(state)
        
        if state.get("error"):
            if trace_context:
                trace_context.update(outputs={"skipped": True, "reason": "previous_error"})
            return state
        
        if "api_url" not in state or not state["api_url"]:
            state["api_url"] = DEFAULT_API_URL
        
        if trace_context:
            trace_context.update(inputs={
                "api_url": state["api_url"],
                "extracted_params": state.get("extracted_params", {})
            })
        
        log_debug(f"Appel API: {state['api_url']}")
        response = requests.get(state["api_url"], timeout=API_TIMEOUT)
        response.raise_for_status()
        
        all_data = response.json()
        
        # Application des filtres
        if state.get("extracted_params") and "filters" in state["extracted_params"]:
            filters = state["extracted_params"]["filters"]
            for key, value in filters.items():
                if key in ["userId", "id"]:
                    all_data = [item for item in all_data if item.get(key) == int(value)]
        
        # Limitation du nombre de résultats
        limit = state["extracted_params"].get("limit", DEFAULT_LIMIT) if state.get("extracted_params") else DEFAULT_LIMIT
        state["api_data"] = all_data[:limit]
        
        if trace_context:
            trace_context.update(outputs={
                "success": True,
                "total_items": len(all_data),
                "filtered_items": len(state["api_data"]),
                "limit_applied": limit
            })
        
        log_debug(f"Données API récupérées: {len(state['api_data'])} éléments")
        
    except Exception as e:
        error_msg = f"Erreur lors de la récupération API: {str(e)}"
        state["error"] = error_msg
        
        if trace_context:
            trace_context.update(outputs={"success": False, "error": error_msg})
        print(f"Erreur: {state['error']}")
    
    finally:
        if trace_context:
            trace_context.__exit__(None, None, None)
    
    return state

def process_data(state: AgentState) -> AgentState:
    """Traite et filtre les données selon les champs demandés"""
    
    trace_context = None
    if langsmith_client:
        try:
            trace_context = langsmith_client.trace(
                name="process_data",
                tags=["processing", "data_transformation"],
                metadata={"step": "3", "component": "data_processor"}
            ).__enter__()
        except:
            pass
    
    try:
        state = ensure_state_keys(state)
        
        if state.get("error") or not state.get("api_data"):
            if trace_context:
                trace_context.update(outputs={"skipped": True, "reason": "no_data_or_error"})
            return state
        
        fields = VALID_API_FIELDS
        if state.get("extracted_params") and "fields" in state["extracted_params"]:
            fields = state["extracted_params"]["fields"]
        
        if trace_context:
            trace_context.update(inputs={
                "raw_data_count": len(state["api_data"]),
                "fields_to_extract": fields
            })
        
        processed_data = []
        for item in state["api_data"]:
            filtered_item = {}
            for field in fields:
                if field in item:
                    filtered_item[field] = item[field]
            processed_data.append(filtered_item)
        
        state["processed_data"] = processed_data
        
        if trace_context:
            trace_context.update(outputs={
                "success": True,
                "processed_items": len(processed_data),
                "extracted_fields": fields
            })
        
        log_debug(f"Données traitées: {len(processed_data)} éléments avec champs {fields}")
        
    except Exception as e:
        error_msg = f"Erreur lors du traitement: {str(e)}"
        state["error"] = error_msg
        
        if trace_context:
            trace_context.update(outputs={"success": False, "error": error_msg})
        print(f"Erreur: {state['error']}")
    
    finally:
        if trace_context:
            trace_context.__exit__(None, None, None)
    
    return state

def create_google_sheet(state: AgentState) -> AgentState:
    """Crée un Google Sheet et y ajoute les données dans un dossier organisé"""
    
    # ✅ CORRECTION : Supprimer le paramètre timeout
    trace_context = create_trace_context(
        name="create_google_sheet",
        tags=["google_sheets", "export"],
        metadata={"step": "4", "component": "sheets_creator"}
    )
    
    try:
        with trace_context or DummyContext():
            state = ensure_state_keys(state)
            
            if state.get("error") or not state.get("processed_data") or not gc:
                if not gc:
                    error_msg = "Google Sheets non configuré"
                    state["error"] = error_msg
                    safe_trace_update(trace_context, outputs={"success": False, "error": error_msg})
                else:
                    safe_trace_update(trace_context, outputs={"skipped": True, "reason": "no_data_or_error"})
                return state
            
            processed_data = state["processed_data"]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sheet_title = f"{SHEETS_DEFAULT_TITLE_PREFIX}_{timestamp}"
            
            safe_trace_update(trace_context, inputs={
                "data_count": len(processed_data),
                "sheet_title": sheet_title,
                "folder_name": SHEETS_FOLDER_NAME
            })
            
            log_debug(f"Création du sheet: {sheet_title}")
            
            # =================================================================
            # 1. SETUP DES CREDENTIALS POUR DRIVE API
            # =================================================================
            folder_id = None
            drive_service = None
            
            try:
                from googleapiclient.discovery import build
                from google.oauth2.service_account import Credentials
                
                log_debug(f"Chargement des credentials depuis: {GOOGLE_CREDENTIALS_PATH}")
                
                # Créer les credentials avec les scopes Drive
                creds = Credentials.from_service_account_file(
                    GOOGLE_CREDENTIALS_PATH, 
                    scopes=GOOGLE_SCOPES
                )
                
                # Créer le service Drive
                drive_service = build('drive', 'v3', credentials=creds)
                log_debug("✅ Service Drive API initialisé")
                
                # =================================================================
                # 2. RECHERCHER/CRÉER LE DOSSIER
                # =================================================================
                log_debug(f"Recherche du dossier '{SHEETS_FOLDER_NAME}'...")
                
                # Rechercher si le dossier existe déjà
                search_query = f"name='{SHEETS_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
                results = drive_service.files().list(
                    q=search_query,
                    fields="files(id, name, parents)"
                ).execute()
                
                folders = results.get('files', [])
                log_debug(f"Dossiers trouvés: {len(folders)}")
                
                if folders:
                    folder_id = folders[0]['id']
                    log_debug(f"✅ Dossier trouvé: {SHEETS_FOLDER_NAME} (ID: {folder_id})")
                else:
                    # Créer le dossier s'il n'existe pas
                    log_debug(f"🔧 Création du dossier '{SHEETS_FOLDER_NAME}'...")
                    folder_metadata = {
                        'name': SHEETS_FOLDER_NAME,
                        'mimeType': 'application/vnd.google-apps.folder'
                    }
                    
                    folder = drive_service.files().create(
                        body=folder_metadata,
                        fields='id'
                    ).execute()
                    
                    folder_id = folder.get('id')
                    log_debug(f"✅ Dossier créé: {SHEETS_FOLDER_NAME} (ID: {folder_id})")
                    
                    # Partager le dossier avec l'email personnel
                    if GOOGLE_PERSONAL_EMAIL:
                        try:
                            permission = {
                                'type': 'user',
                                'role': 'writer',
                                'emailAddress': GOOGLE_PERSONAL_EMAIL
                            }
                            drive_service.permissions().create(
                                fileId=folder_id,
                                body=permission,
                                sendNotificationEmail=False
                            ).execute()
                            log_debug(f"✅ Dossier partagé avec {GOOGLE_PERSONAL_EMAIL}")
                        except Exception as share_error:
                            log_debug(f"⚠️ Erreur partage dossier: {share_error}")
                    
            except ImportError:
                log_debug("❌ google-api-python-client non installé")
                log_debug("📝 Installez avec: pip install google-api-python-client")
                drive_service = None
            except FileNotFoundError:
                log_debug(f"❌ Fichier credentials non trouvé: {GOOGLE_CREDENTIALS_PATH}")
                drive_service = None
            except Exception as drive_error:
                log_debug(f"⚠️ Erreur lors de la configuration Drive API: {drive_error}")
                log_debug("📝 Le sheet sera créé à la racine de Drive")
                drive_service = None
            
            # =================================================================
            # 3. CRÉER LE GOOGLE SHEET
            # =================================================================
            log_debug("Création du Google Sheet...")
            sheet = gc.create(sheet_title)
            sheet_id = sheet.id
            log_debug(f"✅ Sheet créé: {sheet_title} (ID: {sheet_id})")
            
            # =================================================================
            # 4. DÉPLACER LE SHEET DANS LE DOSSIER
            # =================================================================
            if folder_id and drive_service:
                try:
                    log_debug(f"🔧 Déplacement du sheet dans le dossier '{SHEETS_FOLDER_NAME}'...")
                    
                    # Récupérer les parents actuels du fichier
                    file_metadata = drive_service.files().get(
                        fileId=sheet_id, 
                        fields='parents'
                    ).execute()
                    
                    previous_parents = ",".join(file_metadata.get('parents', []))
                    log_debug(f"Parents actuels: {previous_parents}")
                    
                    # Déplacer le fichier vers le nouveau dossier
                    drive_service.files().update(
                        fileId=sheet_id,
                        addParents=folder_id,
                        removeParents=previous_parents,
                        fields='id, parents'
                    ).execute()
                    
                    log_debug(f"✅ Sheet déplacé dans le dossier '{SHEETS_FOLDER_NAME}'")
                    
                    # Vérifier le déplacement
                    updated_file = drive_service.files().get(
                        fileId=sheet_id,
                        fields='parents'
                    ).execute()
                    log_debug(f"Nouveaux parents: {updated_file.get('parents', [])}")
                    
                except Exception as move_error:
                    log_debug(f"⚠️ Erreur lors du déplacement: {move_error}")
                    log_debug("📝 Le sheet reste à la racine mais est utilisable")
            else:
                if not folder_id:
                    log_debug("⚠️ Pas de folder_id - sheet créé à la racine")
                if not drive_service:
                    log_debug("⚠️ Pas de drive_service - sheet créé à la racine")
            
            # =================================================================
            # 5. PARTAGER LE SHEET
            # =================================================================
            if GOOGLE_PERSONAL_EMAIL:
                try:
                    sheet.share(GOOGLE_PERSONAL_EMAIL, perm_type='user', role='writer')
                    log_debug(f"✅ Sheet partagé avec {GOOGLE_PERSONAL_EMAIL}")
                except Exception as share_error:
                    log_debug(f"⚠️ Erreur partage sheet: {share_error}")
            
            # Partage public si configuré
            if SHEETS_SHARE_PUBLICLY:
                try:
                    sheet.share('', perm_type='anyone', role='reader')
                    log_debug("✅ Sheet partagé publiquement en lecture")
                except Exception as public_error:
                    log_debug(f"⚠️ Impossible de partager publiquement: {public_error}")
            
            # =================================================================
            # 6. AJOUTER LES DONNÉES
            # =================================================================
            worksheet = sheet.get_worksheet(0)
            
            if processed_data:
                # En-têtes
                headers = list(processed_data[0].keys())
                worksheet.append_row(headers)
                log_debug(f"✅ En-têtes ajoutés: {headers}")
                
                # Données
                for item in processed_data:
                    row_values = [item.get(header, '') for header in headers]
                    worksheet.append_row(row_values)
                
                log_debug(f"✅ {len(processed_data)} lignes de données ajoutées")
            
            # =================================================================
            # 7. CONSTRUIRE L'URL FINALE
            # =================================================================
            state["sheets_url"] = sheet.url
            
            # Construire l'URL du dossier si disponible
            folder_url = None
            if folder_id:
                folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
                log_debug(f"📁 Dossier Google Drive: {folder_url}")
                log_debug(f"📊 Google Sheet: {sheet.url}")
                log_debug(f"🎯 Le sheet a été organisé dans le dossier '{SHEETS_FOLDER_NAME}'")
            else:
                log_debug(f"📊 Google Sheet (racine Drive): {sheet.url}")
            
            safe_trace_update(trace_context, outputs={
                "success": True,
                "sheet_url": sheet.url,
                "sheet_id": sheet_id,
                "folder_id": folder_id,
                "folder_url": folder_url,
                "rows_added": len(processed_data),
                "moved_to_folder": bool(folder_id and drive_service)
            })
            
            log_debug(f"Google Sheet créé avec succès: {sheet.url}")
            
    except Exception as e:
        error_msg = f"Erreur lors de la création du Google Sheet: {str(e)}"
        state["error"] = error_msg
        
        safe_trace_update(trace_context, outputs={"success": False, "error": error_msg})
        log_debug(f"❌ Erreur: {state['error']}")
        
        # Debug détaillé
        import traceback
        log_debug(f"Stack trace: {traceback.format_exc()}")
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """Génère la réponse finale avec lien vers les stats LangSmith"""
    
    state = ensure_state_keys(state)
    
    if state.get("error"):
        response = f"❌ Erreur: {state['error']}"
    else:
        params = state.get("extracted_params", {})
        
        response = f"""✅ Tâche terminée avec succès !

📊 **Données récupérées:**
- {len(state.get('processed_data', []))} posts traités
- Champs extraits: {', '.join(params.get('fields', ['tous']))}
- Limite appliquée: {params.get('limit', DEFAULT_LIMIT)}

📋 **Google Sheet créé:**
{state.get('sheets_url', 'Non disponible')}

🔗 Vous pouvez maintenant accéder à vos données dans le Google Sheet via le lien ci-dessus."""
        
        # Ajouter le lien LangSmith si disponible
        if langsmith_available:
            project_url = f"https://smith.langchain.com/projects/{LANGSMITH_CONFIG['LANGCHAIN_PROJECT']}"
            response += f"""

💰 **Statistiques détaillées** (tokens, coûts, performance) :
🔗 {project_url}
"""
    
    state["messages"].append(AIMessage(content=response))
    
    # Log optionnel des stats
    if langsmith_available:
        log_langsmith_token_summary()
    
    return state
# =============================================================================
# CONSTRUCTION DU GRAPHE (APRÈS DÉFINITION DES FONCTIONS)
# =============================================================================

def build_graph() -> StateGraph:
    """Construit le graphe LangGraph"""
    
    workflow = StateGraph(AgentState)
    
    # Ajout des nœuds
    workflow.add_node("parse_query", parse_user_query)
    workflow.add_node("fetch_data", fetch_api_data)
    workflow.add_node("process_data", process_data)
    workflow.add_node("create_sheet", create_google_sheet)
    workflow.add_node("respond", generate_response)
    
    # Définition des connexions
    workflow.add_edge(START, "parse_query")
    workflow.add_edge("parse_query", "fetch_data")
    workflow.add_edge("fetch_data", "process_data")
    workflow.add_edge("process_data", "create_sheet")
    workflow.add_edge("create_sheet", "respond")
    workflow.add_edge("respond", END)
    
    return workflow.compile()

# Instance du graphe pour l'export
graph = build_graph()

# =============================================================================
# UTILITAIRES D'ÉTAT
# =============================================================================

def get_initial_state() -> AgentState:
    """Configuration de l'état initial par défaut"""
    return {
        "messages": [],
        "api_url": DEFAULT_API_URL,
        "user_query": "",
        "extracted_params": None,
        "api_data": None,
        "processed_data": None,
        "sheets_url": "",
        "error": ""
    }

# =============================================================================
# FONCTION D'EXÉCUTION AVEC TRACING GLOBAL
# =============================================================================

def run_agent_with_tracing(user_input: str, run_name: str = None) -> AgentState:
    """Exécute l'agent avec un tracing global de la session"""
    
    if run_name is None:
        run_name = f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    trace_context = None
    if langsmith_client:
        try:
            trace_context = langsmith_client.trace(
                name=run_name,
                tags=["agent_execution", "full_pipeline"],
                metadata={"user_input": user_input}
            ).__enter__()
        except:
            pass
    
    try:
        # État initial
        initial_state = get_initial_state()
        initial_state["messages"] = [HumanMessage(content=user_input)]
        
        if trace_context:
            trace_context.update(inputs={"user_input": user_input})
        
        log_debug(f"Démarrage de l'agent avec input: {user_input}")
        
        # Exécution du graphe
        result = graph.invoke(initial_state)
        
        if trace_context:
            trace_context.update(outputs={
                "success": True,
                "final_state": {
                    "sheets_url": result.get("sheets_url"),
                    "processed_data_count": len(result.get("processed_data", [])),
                    "error": result.get("error")
                }
            })
        
        return result
        
    except Exception as e:
        error_msg = f"Erreur lors de l'exécution de l'agent: {str(e)}"
        if trace_context:
            trace_context.update(outputs={"success": False, "error": error_msg})
        print(f"❌ {error_msg}")
        raise
    
    finally:
        if trace_context:
            trace_context.__exit__(None, None, None)

# =============================================================================
# FONCTION DE TEST PRINCIPALE
# =============================================================================

def main():
    """Fonction principale pour tester l'agent"""
    try:
        print("🚀 Démarrage de l'agent API to Sheets")
        print(f"📋 Configuration:")
        print(f"   - Modèle OpenAI: {OPENAI_MODEL}")
        print(f"   - API par défaut: {DEFAULT_API_URL}")
        print(f"   - Limite par défaut: {DEFAULT_LIMIT}")
        print(f"   - LangSmith activé: {'✅' if langsmith_client else '❌'}")
        print(f"   - Debug activé: {'✅' if DEBUG else '❌'}")
        print(f"   - Environnement valide: {'✅' if env_valid else '❌'}")
        print()
        
        if not env_valid:
            print("⚠️ Configuration incomplète - certaines fonctionnalités peuvent ne pas fonctionner")
        
        # Test avec une requête exemple
        test_query = "récupère 5 posts avec title et id"
        print(f"🧪 Test avec la requête: '{test_query}'")
        
        result = run_agent_with_tracing(test_query, "test_execution")
        
        if result.get("error"):
            print(f"❌ Erreur: {result['error']}")
        else:
            print("✅ Exécution réussie!")
            if result.get("sheets_url"):
                print(f"📊 Google Sheet créé: {result['sheets_url']}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return 1
    
    return 0

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    exit(main())

# =============================================================================
# EXPORT POUR LANGGRAPH STUDIO
# =============================================================================

# Ces variables et fonctions sont exportées pour LangGraph Studio
__all__ = [
    'graph',
    'AgentState', 
    'get_initial_state',
    'run_agent_with_tracing',
    'parse_user_query',
    'fetch_api_data',
    'process_data', 
    'create_google_sheet',
    'generate_response'
]

# =============================================================================
# CONFIGURATION LANGGRAPH STUDIO
# =============================================================================

# Metadata pour LangGraph Studio
GRAPH_METADATA = {
    "title": "API to Google Sheets Agent",
    "description": "Agent qui récupère des données d'APIs et les exporte vers Google Sheets",
    "version": "1.0.0",
    "author": "Your Name",
    "tags": ["api", "google-sheets", "data-export", "langgraph"],
    "requirements": [
        "openai", 
        "gspread", 
        "google-auth",
        "requests",
        "langchain-openai",
        "langgraph",
        "langsmith"
    ],
    "configuration": {
        "required_env_vars": [
            "OPENAI_API_KEY",
            "GOOGLE_CREDENTIALS_PATH"
        ],
        "optional_env_vars": [
            "LANGSMITH_API_KEY",
            "GOOGLE_PERSONAL_EMAIL",
            "DEFAULT_API_URL",
            "DEBUG"
        ]
    }
}

# Configuration par défaut pour les tests dans Studio
DEFAULT_TEST_CONFIG = {
    "user_input": "récupère 5 posts avec title et id",
    "api_url": DEFAULT_API_URL,
    "expected_fields": ["title", "id"],
    "expected_limit": 5
}

# =============================================================================
# CONFIGURATION DE DÉMARRAGE POUR LANGGRAPH STUDIO
# =============================================================================

print(f"🔧 LangGraph Agent chargé - Configuration:")
print(f"   - OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")
print(f"   - Google Sheets: {'✅' if gc else '❌'}")
print(f"   - LangSmith: {'✅' if langsmith_client else '❌'}")
print(f"   - Debug: {'✅' if DEBUG else '❌'}")

if not env_valid:
    print("⚠️ Configuration incomplète - vérifiez votre fichier .env")
    print("📝 Variables requises:")
    print("   - OPENAI_API_KEY")
    print("   - GOOGLE_CREDENTIALS_PATH (fichier google-credentials.json)")
    print("📝 Variables optionnelles:")
    print("   - LANGSMITH_API_KEY")
    print("   - GOOGLE_PERSONAL_EMAIL")
else:
    print("✅ Agent prêt à être utilisé !")

# Afficher un résumé de la configuration
print("\n📋 Résumé de la configuration:")
print(f"   - Modèle OpenAI: {OPENAI_MODEL}")
print(f"   - Temperature: {OPENAI_TEMPERATURE}")
print(f"   - API timeout: {API_TIMEOUT}s")
print(f"   - Limite par défaut: {DEFAULT_LIMIT} posts")
print(f"   - Limite max: {MAX_LIMIT} posts")
print(f"   - URL API par défaut: {DEFAULT_API_URL}")
print(f"   - Dossier Google Sheets: {SHEETS_FOLDER_NAME}")
print(f"   - Préfixe des sheets: {SHEETS_DEFAULT_TITLE_PREFIX}")

if GOOGLE_PERSONAL_EMAIL:
    print(f"   - Email personnel: {GOOGLE_PERSONAL_EMAIL}")

if langsmith_client:
    print(f"   - Projet LangSmith: {LANGSMITH_CONFIG.get('LANGCHAIN_PROJECT', 'N/A')}")

print("\n🔗 Pour tester l'agent:")
print('   result = run_agent_with_tracing("récupère 5 posts avec title et id")')
print("   print(result)")

print("\n🎯 Agent prêt pour LangGraph Studio !")