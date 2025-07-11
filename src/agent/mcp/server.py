#!/usr/bin/env python3
"""
Serveur MCP complet avec Agent LangGraph
"""

import asyncio
import sys
import json
import requests
import os
from pathlib import Path
from typing import List, Dict, Any

# Ajouter le path du projet
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def send_message(message: dict):
    json_str = json.dumps(message)
    print(json_str, flush=True)

def log_to_stderr(message: str):
    print(f"[DEBUG] {message}", file=sys.stderr, flush=True)

# Import de l'agent LangGraph
try:
    # Rediriger stdout temporairement pour éviter les interférences
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    
    from agent import graph as agent_module
    
    # Restaurer stdout
    sys.stdout = original_stdout
    
    AGENT_AVAILABLE = True
    log_to_stderr("✅ Agent LangGraph importé avec succès")
    
    # Vérifier les fonctions principales
    required_functions = ['run_agent_with_tracing', 'get_initial_state', 'parse_user_query', 'graph']
    available_functions = [f for f in required_functions if hasattr(agent_module, f)]
    
    log_to_stderr(f"✅ {len(available_functions)}/{len(required_functions)} fonctions agent disponibles")
    
    # Variables de configuration
    config_vars = {
        'OPENAI_API_KEY': getattr(agent_module, 'OPENAI_API_KEY', None),
        'DEFAULT_API_URL': getattr(agent_module, 'DEFAULT_API_URL', None),
        'DEFAULT_LIMIT': getattr(agent_module, 'DEFAULT_LIMIT', 10),
        'gc': getattr(agent_module, 'gc', None),
    }
    
    log_to_stderr("✅ Configuration agent chargée")
    
except Exception as e:
    sys.stdout = original_stdout
    AGENT_AVAILABLE = False
    agent_module = None
    available_functions = []
    config_vars = {}
    log_to_stderr(f"❌ Agent LangGraph non disponible: {e}")

# Test d'import Google Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GOOGLE_SHEETS_AVAILABLE = True
    log_to_stderr("✅ Google Sheets importé")
except ImportError as e:
    GOOGLE_SHEETS_AVAILABLE = False
    log_to_stderr(f"❌ Google Sheets non disponible: {e}")

def check_google_credentials():
    """Vérifie si les credentials Google sont disponibles"""
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "google-credentials.json")
    full_path = project_root / credentials_path
    return full_path.exists()

def make_api_request(endpoint: str, limit: int = 10) -> List[Dict]:
    """Requête API simple vers JSONPlaceholder"""
    try:
        url = f"https://jsonplaceholder.typicode.com/{endpoint}"
        response = requests.get(url, params={'_limit': limit}, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list):
            return data[:limit]
        elif isinstance(data, dict):
            return [data]
        else:
            return []
            
    except Exception as e:
        log_to_stderr(f"Erreur API {endpoint}: {e}")
        return []

def run_agent_safely(query: str) -> dict:
    """Exécute l'agent LangGraph de manière sécurisée"""
    if not AGENT_AVAILABLE:
        return {"error": "Agent LangGraph non disponible"}
    
    try:
        # Rediriger stdout pendant l'exécution de l'agent
        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        
        log_to_stderr(f"🤖 Exécution agent avec: {query}")
        
        # Exécuter l'agent
        run_agent_func = getattr(agent_module, 'run_agent_with_tracing')
        result = run_agent_func(query)
        
        # Restaurer stdout
        sys.stdout = original_stdout
        
        log_to_stderr("✅ Agent exécuté avec succès")
        
        return {"success": True, "result": result}
        
    except Exception as e:
        sys.stdout = original_stdout
        log_to_stderr(f"❌ Erreur agent: {e}")
        return {"error": str(e)}

async def handle_request(request: dict) -> dict:
    """Traite une requête MCP"""
    method = request.get("method")
    request_id = request.get("id")
    
    log_to_stderr(f"Requête reçue: {method}")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "api-sheets-agent",
                    "version": "1.0.0"
                }
            }
        }
    
    elif method == "notifications/initialized":
        log_to_stderr("Initialized notification reçue")
        return None
    
    elif method == "tools/list":
        tools = [
            {
                "name": "hello",
                "description": "Test de connexion MCP avec status complet",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_posts",
                "description": "Récupère des posts depuis JSONPlaceholder",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Nombre de posts (1-20)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_users",
                "description": "Récupère des utilisateurs depuis JSONPlaceholder",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Nombre d'utilisateurs (1-10)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": []
                }
            }
        ]
        
        # Ajouter l'outil agent si disponible
        if AGENT_AVAILABLE and 'run_agent_with_tracing' in available_functions:
            tools.append({
                "name": "run_agent",
                "description": "Exécute l'agent LangGraph complet pour traiter une requête complexe (API + Google Sheets)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Requête à traiter par l'agent (ex: 'récupère 5 posts et sauvegarde dans une feuille')"
                        }
                    },
                    "required": ["query"]
                }
            })
        
        # Ajouter les outils Google Sheets si disponibles
        if GOOGLE_SHEETS_AVAILABLE and check_google_credentials():
            tools.append({
                "name": "create_sheet",
                "description": "Crée une feuille Google Sheets simple",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Titre de la feuille"
                        }
                    },
                    "required": ["title"]
                }
            })
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools}
        }
    
    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        log_to_stderr(f"Appel outil: {tool_name} avec {arguments}")
        
        if tool_name == "hello":
            agent_status = "✅ Disponible" if AGENT_AVAILABLE else "❌ Non disponible"
            google_status = "✅ Configuré" if (GOOGLE_SHEETS_AVAILABLE and check_google_credentials()) else "❌ Non configuré"
            
            content = f"""🎉 **SERVEUR MCP COMPLET OPÉRATIONNEL !**

✅ **Serveur MCP:** Opérationnel
🤖 **Agent LangGraph:** {agent_status}
📊 **Google Sheets:** {google_status}
🌐 **APIs externes:** JSONPlaceholder disponible

📁 **Projet:** {project_root.name}

💡 **Commandes disponibles:**
- `get_posts limit=3` - Récupérer des posts
- `get_users limit=3` - Récupérer des utilisateurs
{'- `run_agent query="récupère 5 posts et sauvegarde dans une feuille"` - Agent complet !' if AGENT_AVAILABLE else ''}
{'- `create_sheet title="Test"` - Créer une feuille simple' if (GOOGLE_SHEETS_AVAILABLE and check_google_credentials()) else ''}

🚀 **Agent LangGraph intégré:** Pipeline complet API → Google Sheets disponible !"""
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": content}]
                }
            }
        
        elif tool_name == "get_posts":
            limit = arguments.get("limit", 5)
            posts = make_api_request("posts", limit=limit)
            
            if not posts:
                content = "❌ Impossible de récupérer les posts"
            else:
                content = f"📝 **{len(posts)} posts récupérés:**\n\n"
                for i, post in enumerate(posts, 1):
                    content += f"**{i}. Post {post.get('id')}**\n"
                    content += f"   📝 {post.get('title', '')[:60]}...\n"
                    content += f"   👤 User ID: {post.get('userId', 'N/A')}\n\n"
                
                if AGENT_AVAILABLE:
                    content += f"\n💡 **Astuce:** Utilisez `run_agent query=\"sauvegarde ces {len(posts)} posts dans une feuille Google Sheets\"` pour les exporter automatiquement !"
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": content}]
                }
            }
        
        elif tool_name == "get_users":
            limit = arguments.get("limit", 5)
            users = make_api_request("users", limit=limit)
            
            if not users:
                content = "❌ Impossible de récupérer les utilisateurs"
            else:
                content = f"👥 **{len(users)} utilisateurs récupérés:**\n\n"
                for i, user in enumerate(users, 1):
                    content += f"**{i}. {user.get('name')}**\n"
                    content += f"   📧 {user.get('email')}\n"
                    content += f"   🌐 {user.get('website', 'Pas de site')}\n\n"
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": content}]
                }
            }
        
        elif tool_name == "run_agent":
            query = arguments.get("query", "")
            
            if not query:
                content = "❌ Veuillez fournir une requête pour l'agent"
            else:
                result = run_agent_safely(query)
                
                if result.get("success"):
                    agent_result = result["result"]
                    
                    if isinstance(agent_result, dict):
                        final_answer = agent_result.get('final_answer', str(agent_result))
                        sheets_url = agent_result.get('sheets_url', '')
                        
                        content = f"""🤖 **AGENT LANGGRAPH EXÉCUTÉ AVEC SUCCÈS !**

📋 **Résultat:**
{final_answer}"""
                        
                        if sheets_url:
                            content += f"""

🔗 **Feuille Google Sheets créée:**
{sheets_url}

✅ **Pipeline complet réalisé:** API → Traitement → Google Sheets"""
                    else:
                        content = f"🤖 **Résultat de l'agent:**\n\n{str(agent_result)}"
                        
                else:
                    content = f"❌ **Erreur de l'agent:** {result.get('error')}"
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": content}]
                }
            }
        
        elif tool_name == "create_sheet":
            content = "❌ Fonctionnalité Google Sheets en cours de correction (problème OAuth)\n\n💡 **Alternative:** Utilisez `run_agent` qui inclut la création de feuilles avec votre agent LangGraph complet !"
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": content}]
                }
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Outil inconnu: {tool_name}"
                }
            }
    
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": "Méthode non trouvée"
            }
        }

async def main():
    """Boucle principale du serveur"""
    log_to_stderr("🚀 Serveur MCP COMPLET avec Agent LangGraph démarré")
    log_to_stderr(f"📁 Projet: {project_root}")
    log_to_stderr(f"🤖 Agent: {'✅' if AGENT_AVAILABLE else '❌'}")
    log_to_stderr(f"📊 Google Sheets: {'✅' if (GOOGLE_SHEETS_AVAILABLE and check_google_credentials()) else '❌'}")
    
    try:
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    log_to_stderr(f"Requête parsée: {request.get('method')}")
                except json.JSONDecodeError as e:
                    log_to_stderr(f"Erreur JSON: {e}")
                    continue
                
                response = await handle_request(request)
                
                if response:
                    send_message(response)
                    log_to_stderr(f"Réponse envoyée pour: {request.get('method')}")
                
            except Exception as e:
                log_to_stderr(f"Erreur lors du traitement: {e}")
                continue
                
    except KeyboardInterrupt:
        log_to_stderr("Serveur arrêté")
    except Exception as e:
        log_to_stderr(f"Erreur fatale: {e}")

if __name__ == "__main__":
    asyncio.run(main())