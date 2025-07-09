import json
import requests
import os
from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
import re

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# Configuration via variables d'environnement
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "./google-credentials.json")

# État typé pour LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    api_url: str
    user_query: str
    extracted_params: Optional[Dict[str, Any]]
    api_data: Optional[List[Dict]]
    processed_data: Optional[List[Dict]]
    sheets_url: str
    error: str

# Configuration globale
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.1
)

# Configuration Google Sheets
def setup_google_sheets():
    """Configuration de l'accès Google Sheets"""
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_PATH, scopes=scopes
        )
        return gspread.authorize(creds)
    except Exception as e:
        print(f"Erreur configuration Google Sheets: {e}")
        return None

gc = setup_google_sheets()

def parse_user_query(state: AgentState) -> AgentState:
    """Parse la requête utilisateur pour extraire les paramètres"""
    
    print("🚀 VERSION MISE À JOUR CHARGÉE - TIMESTAMP:", datetime.now())
    
    # Récupérer la dernière requête utilisateur - GESTION DES DICTIONNAIRES
    user_query = ""
    messages = state.get("messages", [])
    
    for message in reversed(messages):
        # Gestion des messages sous forme de dictionnaire (LangGraph Studio)
        if isinstance(message, dict):
            if message.get('type') == 'human' and message.get('content'):
                user_query = message['content']
                break
        # Gestion des objets HumanMessage (tests locaux)
        elif isinstance(message, HumanMessage):
            user_query = message.content
            break
    
    print(f"📝 Requête à analyser: '{user_query}'")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un extracteur de paramètres ULTRA-PRÉCIS. Tu dois analyser la requête utilisateur et extraire EXACTEMENT ce qui est demandé.

        **CHAMPS DISPONIBLES UNIQUEMENT:**
        - userId
        - id  
        - title
        - body

        **INSTRUCTIONS CRITIQUES:**

        🔢 **NOMBRE DE POSTS:**
        - Cherche le premier nombre dans la requête
        - "récupère 5 posts" → limit: 5
        - "donne-moi 3 posts" → limit: 3
        - "prends 7 posts" → limit: 7
        - Si AUCUN nombre → limit: 10

        📝 **CHAMPS À EXTRAIRE:**
        - Si l'utilisateur mentionne des champs spécifiques → SEULEMENT ces champs
        - Si "avec title et id" → ["title", "id"] (PAS les autres)
        - Si "seulement title" → ["title"]
        - Si "userId et body" → ["userId", "body"]
        - Si aucun champ spécifique → ["userId", "id", "title", "body"]

        **MOTS-CLÉS IMPORTANTS:**
        - "avec" = champs spécifiques suivent
        - "seulement", "uniquement" = restriction aux champs mentionnés
        - "et" = séparateur entre champs

        **ANALYSE OBLIGATOIRE:**
        1. Trouve le nombre → limit
        2. Trouve les champs mentionnés → fields
        3. Construis le JSON

        **EXEMPLES CRITIQUES:**
        
        "récupère 5 posts avec title et id"
        → Nombre: 5, Champs: title et id
        → {{"limit": 5, "fields": ["title", "id"], "filters": {{}}, "description": "5 posts avec title et id"}}

        "donne-moi 3 posts avec userId"
        → Nombre: 3, Champs: userId
        → {{"limit": 3, "fields": ["userId"], "filters": {{}}, "description": "3 posts avec userId"}}

        "récupère 7 posts"
        → Nombre: 7, Champs: aucun spécifique
        → {{"limit": 7, "fields": ["userId", "id", "title", "body"], "filters": {{}}, "description": "7 posts avec tous les champs"}}

        **RÉPONSE OBLIGATOIRE:**
        JSON uniquement, sans texte supplémentaire:
        {{
            "limit": nombre_trouvé,
            "fields": ["champs_mentionnés"],
            "filters": {{}},
            "description": "description"
        }}
        """),
        ("human", "REQUÊTE À ANALYSER: {query}\n\nRéponds UNIQUEMENT avec le JSON, sans explication:")
    ])
    
    try:
        response = llm.invoke(prompt.format(query=user_query))
        print(f"🤖 Réponse LLM brute: {response.content}")
        
        # Extraire le JSON de la réponse
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group())
            print(f"📊 Paramètres LLM: {params}")
            
            # Validation et nettoyage des paramètres
            validated_params = validate_extracted_params(params, user_query)
            
            state["extracted_params"] = validated_params
            state["user_query"] = user_query
            print(f"✅ Paramètres finaux: {validated_params}")
        else:
            raise ValueError("Impossible d'extraire les paramètres JSON")
            
    except Exception as e:
        state["error"] = f"Erreur lors du parsing: {str(e)}"
        print(f"❌ Erreur: {state['error']}")
    
    return state

def validate_extracted_params(params: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """Valide et corrige AGRESSIVEMENT les paramètres extraits"""
    
    user_query_lower = user_query.lower()
    print(f"🔍 Validation pour: '{user_query_lower}'")
    
    # 1. VALIDATION DU LIMIT (priorité absolue)
    numbers = re.findall(r'\b(\d+)\b', user_query)
    if numbers:
        # Prendre le premier nombre trouvé
        params["limit"] = int(numbers[0])
        print(f"🔧 Limite corrigée: {params['limit']} (trouvé dans '{user_query}')")
    elif "limit" not in params or not isinstance(params["limit"], int) or params["limit"] <= 0:
        params["limit"] = 10
        print(f"🔧 Limite par défaut: {params['limit']}")
    
    # 2. VALIDATION DES FIELDS (analyse manuelle des mots-clés)
    valid_fields = ["userId", "id", "title", "body"]
    
    # Recherche de mots-clés de champs spécifiques
    field_keywords = {
        "title": ["title", "titre"],
        "id": ["id", "identifiant"],
        "userId": ["userid", "user", "utilisateur"],
        "body": ["body", "contenu", "texte"]
    }
    
    # Détecter si des champs spécifiques sont mentionnés
    mentioned_fields = []
    for field, keywords in field_keywords.items():
        if any(f" {keyword} " in f" {user_query_lower} " or 
               user_query_lower.startswith(keyword + " ") or 
               user_query_lower.endswith(" " + keyword) for keyword in keywords):
            mentioned_fields.append(field)
    
    print(f"🔍 Champs détectés: {mentioned_fields}")
    
    # Détecter les mots-clés de restriction
    restriction_keywords = ["avec", "seulement", "uniquement", "juste"]
    has_restriction_keywords = any(word in user_query_lower for word in restriction_keywords)
    print(f"🔍 Mots de restriction trouvés: {has_restriction_keywords}")
    
    if mentioned_fields and has_restriction_keywords:
        # Si des champs spécifiques sont mentionnés avec restriction
        params["fields"] = mentioned_fields
        print(f"🔧 Champs corrigés (restriction): {params['fields']} (détectés dans '{user_query}')")
    elif not mentioned_fields:
        # Si aucun champ spécifique mentionné
        params["fields"] = valid_fields
        print(f"🔧 Champs par défaut (aucun spécifique): {params['fields']}")
    else:
        # Validation classique si les paramètres semblent cohérents
        if "fields" not in params or not isinstance(params["fields"], list):
            params["fields"] = valid_fields
        else:
            # Filtrer les champs valides
            original_fields = params["fields"][:]
            params["fields"] = [field for field in params["fields"] if field in valid_fields]
            if not params["fields"]:
                params["fields"] = valid_fields
            print(f"🔧 Champs validés: {original_fields} → {params['fields']}")
    
    # 3. VALIDATION DES FILTERS
    if "filters" not in params or not isinstance(params["filters"], dict):
        params["filters"] = {}
    
    # 4. VALIDATION DE LA DESCRIPTION
    if "description" not in params:
        params["description"] = f"Récupération de {params['limit']} posts avec les champs {', '.join(params['fields'])}"
    
    return params


def fetch_api_data(state: AgentState) -> AgentState:
    """Récupère les données depuis l'API"""
    
    # S'assurer que l'état a toutes les clés nécessaires
    state = ensure_state_keys(state)
    
    print(f"🔍 fetch_api_data - État reçu: {list(state.keys())}")
    print(f"🔍 api_url: {state.get('api_url', 'NON DÉFINI')}")
    
    if state.get("error"):
        return state
    
    # Initialiser api_url si pas présent
    if "api_url" not in state or not state["api_url"]:
        state["api_url"] = "https://jsonplaceholder.typicode.com/posts"
        print(f"🔧 api_url initialisé: {state['api_url']}")
    
    try:
        # Récupération des données
        print(f"🌐 Appel API: {state['api_url']}")
        response = requests.get(state["api_url"])
        response.raise_for_status()
        
        all_data = response.json()
        
        # Application des filtres
        if state.get("extracted_params") and "filters" in state["extracted_params"]:
            filters = state["extracted_params"]["filters"]
            for key, value in filters.items():
                if key in ["userId", "id"]:
                    all_data = [item for item in all_data if item.get(key) == int(value)]
        
        # Limitation du nombre de résultats
        limit = state["extracted_params"].get("limit", 10) if state.get("extracted_params") else 10
        state["api_data"] = all_data[:limit]
        
        print(f"Données API récupérées: {len(state['api_data'])} éléments")
        
    except Exception as e:
        state["error"] = f"Erreur lors de la récupération API: {str(e)}"
        print(f"Erreur: {state['error']}")
    
    return state

def process_data(state: AgentState) -> AgentState:
    """Traite et filtre les données selon les champs demandés"""
    
    # S'assurer que l'état a toutes les clés nécessaires
    state = ensure_state_keys(state)
    
    if state.get("error") or not state.get("api_data"):
        return state
    
    try:
        # Champs à extraire
        fields = ["userId", "id", "title", "body"]  # par défaut
        if state.get("extracted_params") and "fields" in state["extracted_params"]:
            fields = state["extracted_params"]["fields"]
        
        # Filtrage des données
        processed_data = []
        for item in state["api_data"]:
            filtered_item = {}
            for field in fields:
                if field in item:
                    filtered_item[field] = item[field]
            processed_data.append(filtered_item)
        
        state["processed_data"] = processed_data
        print(f"Données traitées: {len(processed_data)} éléments avec champs {fields}")
        
    except Exception as e:
        state["error"] = f"Erreur lors du traitement: {str(e)}"
        print(f"Erreur: {state['error']}")
    
    return state


def create_google_sheet(state: AgentState) -> AgentState:
    """Crée un Google Sheet et y ajoute les données dans un dossier organisé"""
    
    # S'assurer que l'état a toutes les clés nécessaires
    state = ensure_state_keys(state)
    
    if state.get("error") or not state.get("processed_data") or not gc:
        if not gc:
            state["error"] = "Google Sheets non configuré"
        return state
    
    try:
        processed_data = state["processed_data"]
        
        # Création du Google Sheet avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sheet_title = f"API_Data_{timestamp}"
        
        print(f"🔍 Création du sheet: {sheet_title}")
        
        # 1. SETUP CORRECT DES CREDENTIALS POUR DRIVE API
        folder_id = None
        drive_service = None
        
        try:
            from googleapiclient.discovery import build
            from google.oauth2.service_account import Credentials
            import os
            
            # IMPORTANT: Recréer les credentials directement depuis le fichier
            GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "./google-credentials.json")
            
            print(f"🔧 Chargement des credentials depuis: {GOOGLE_CREDENTIALS_PATH}")
            
            # Définir les scopes nécessaires
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Créer les credentials depuis le fichier JSON
            creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=scopes)
            
            # Créer le service Drive
            drive_service = build('drive', 'v3', credentials=creds)
            
            print(f"✅ Service Drive API initialisé")
            
            # 2. RECHERCHER/CRÉER LE DOSSIER
            folder_name = "API_Data_Exports"
            print(f"🔍 Recherche du dossier '{folder_name}'...")
            
            # Rechercher si le dossier existe déjà
            results = drive_service.files().list(
                q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)"
            ).execute()
            
            folders = results.get('files', [])
            
            if folders:
                folder_id = folders[0]['id']
                print(f"✅ Dossier trouvé: {folder_name} (ID: {folder_id})")
            else:
                # Créer le dossier s'il n'existe pas
                print(f"🔧 Création du dossier '{folder_name}'...")
                folder_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = drive_service.files().create(
                    body=folder_metadata, 
                    fields='id'
                ).execute()
                folder_id = folder.get('id')
                print(f"✅ Dossier créé: {folder_name} (ID: {folder_id})")
                
                # PARTAGER LE DOSSIER avec votre compte personnel
                PERSONAL_EMAIL = "personal e-mail"  # ← Remplacez par votre email
                if PERSONAL_EMAIL:
                    try:
                        permission = {
                            'type': 'user',
                            'role': 'writer',
                            'emailAddress': PERSONAL_EMAIL
                        }
                        drive_service.permissions().create(
                            fileId=folder_id,
                            body=permission,
                            sendNotificationEmail=True
                        ).execute()
                        print(f"✅ Dossier partagé avec {PERSONAL_EMAIL}")
                    except Exception as e:
                        print(f"⚠️ Erreur partage dossier: {e}")
                        
        except ImportError:
            print("❌ google-api-python-client non installé")
            print("📝 Installez avec: pip install google-api-python-client")
            drive_service = None
        except FileNotFoundError:
            print(f"❌ Fichier credentials non trouvé: {GOOGLE_CREDENTIALS_PATH}")
            drive_service = None
        except Exception as e:
            print(f"⚠️ Erreur lors de la configuration Drive API: {e}")
            print("📝 Le sheet sera créé à la racine de Drive")
            drive_service = None
        
        # 3. CRÉER LE GOOGLE SHEET
        sheet = gc.create(sheet_title)
        sheet_id = sheet.id
        print(f"✅ Sheet créé: {sheet_title} (ID: {sheet_id})")
        
        # 4. DÉPLACER LE SHEET DANS LE DOSSIER
        if folder_id and drive_service:
            try:
                print(f"🔧 Déplacement du sheet dans le dossier...")
                
                # Récupérer les parents actuels du fichier
                file = drive_service.files().get(fileId=sheet_id, fields='parents').execute()
                previous_parents = ",".join(file.get('parents', []))
                
                # Déplacer le fichier vers le nouveau dossier
                drive_service.files().update(
                    fileId=sheet_id,
                    addParents=folder_id,
                    removeParents=previous_parents,
                    fields='id, parents'
                ).execute()
                
                print(f"✅ Sheet déplacé dans le dossier '{folder_name}'")
                
            except Exception as e:
                print(f"⚠️ Erreur lors du déplacement: {e}")
                print("📝 Le sheet reste à la racine mais est utilisable")
        
        # 5. PARTAGER LE SHEET avec votre compte personnel
        PERSONAL_EMAIL = "personal e-mail"  # ← Remplacez par votre email
        if PERSONAL_EMAIL:
            try:
                sheet.share(PERSONAL_EMAIL, perm_type='user', role='writer')
                print(f"✅ Sheet partagé avec {PERSONAL_EMAIL}")
            except Exception as e:
                print(f"⚠️ Erreur partage sheet: {e}")
        
        # 6. PARTAGER PUBLIQUEMENT (optionnel)
        try:
            sheet.share('', perm_type='anyone', role='reader')
            print("✅ Sheet partagé publiquement en lecture")
        except Exception as e:
            print(f"⚠️ Impossible de partager publiquement: {e}")
        
        # 7. ACCÈS À LA PREMIÈRE FEUILLE ET AJOUT DES DONNÉES
        worksheet = sheet.get_worksheet(0)
        
        # Ajout des données
        if processed_data:
            # En-têtes (récupération des clés du premier élément)
            headers = list(processed_data[0].keys())
            worksheet.append_row(headers)
            print(f"✅ En-têtes ajoutés: {headers}")
            
            # Données (conversion de chaque dictionnaire en liste de valeurs)
            for item in processed_data:
                row_values = [item.get(header, '') for header in headers]
                worksheet.append_row(row_values)
            
            print(f"✅ {len(processed_data)} lignes de données ajoutées")
        
        # 8. CONSTRUIRE L'URL FINALE ET AFFICHAGE
        state["sheets_url"] = sheet.url
        
        if folder_id:
            folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
            print(f"📁 Dossier Google Drive: {folder_url}")
            print(f"📊 Google Sheet: {sheet.url}")
            print(f"🎯 Le sheet a été organisé dans le dossier '{folder_name}'")
        else:
            print(f"📊 Google Sheet (racine Drive): {sheet.url}")
        
    except Exception as e:
        state["error"] = f"Erreur lors de la création du Google Sheet: {str(e)}"
        print(f"❌ Erreur: {state['error']}")
    
    return state


def generate_response(state: AgentState) -> AgentState:
    """Génère la réponse finale"""
    
    # S'assurer que l'état a toutes les clés nécessaires
    state = ensure_state_keys(state)
    
    if state.get("error"):
        response = f"❌ Erreur: {state['error']}"
    else:
        params = state.get("extracted_params", {})
        response = f"""✅ Tâche terminée avec succès !

📊 **Données récupérées:**
- {len(state.get('processed_data', []))} posts traités
- Champs extraits: {', '.join(params.get('fields', ['tous']))}
- Limite appliquée: {params.get('limit', 10)}

📋 **Google Sheet créé:**
{state.get('sheets_url', 'Non disponible')}

🔗 Vous pouvez maintenant accéder à vos données dans le Google Sheet via le lien ci-dessus."""
    
    state["messages"].append(AIMessage(content=response))
    return state

# Construction du graphe
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

# Configuration de l'état initial par défaut avec toutes les valeurs
def get_initial_state() -> AgentState:
    return {
        "messages": [],
        "api_url": "https://jsonplaceholder.typicode.com/posts",
        "user_query": "",
        "extracted_params": None,
        "api_data": None,
        "processed_data": None,
        "sheets_url": "",
        "error": ""
    }

# Fonction pour initialiser l'état si des clés manquent
def ensure_state_keys(state: AgentState) -> AgentState:
    """S'assurer que toutes les clés nécessaires sont présentes dans l'état"""
    default_state = get_initial_state()
    
    for key, default_value in default_state.items():
        if key not in state:
            state[key] = default_value
    
    return state