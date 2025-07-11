"""
Utilitaires de formatage pour les réponses MCP
"""

import json
from typing import Dict, Any, List

def format_success_response(result: Dict[str, Any]) -> str:
    """Formate une réponse de succès pour MCP"""
    
    params = result.get("extracted_params", {})
    processed_data = result.get("processed_data", [])
    sheets_url = result.get("sheets_url", "Non disponible")
    
    return f"""✅ Tâche terminée avec succès !

📊 **Données récupérées:**
- {len(processed_data)} éléments traités
- Champs extraits: {', '.join(params.get('fields', ['tous']))}
- Limite appliquée: {params.get('limit', 10)}

📋 **Google Sheet créé:**
{sheets_url}

🔗 Votre export est prêt !"""

def format_validation_response(params: Dict[str, Any], query: str) -> str:
    """Formate une réponse de validation pour MCP"""
    
    return f"""✅ Requête valide !

📋 **Paramètres extraits:**
- Limite: {params.get('limit', 10)}
- Champs: {', '.join(params.get('fields', ['tous']))}
- Filtres: {json.dumps(params.get('filters', {}), indent=2)}
- Description: {params.get('description', 'Non disponible')}

🎯 **Requête originale:** "{query}"
"""

def format_status_response(status_data: Dict[str, Any]) -> str:
    """Formate une réponse de statut pour MCP"""
    
    return f"""🤖 **Statut de l'agent API to Sheets**

🔧 **Configuration:**
- Modèle OpenAI: {status_data.get('model', 'N/A')}
- OpenAI configuré: {status_data.get('openai_status', '❌')}
- Google Sheets configuré: {status_data.get('sheets_status', '❌')}
- LangSmith configuré: {status_data.get('langsmith_status', '❌')}

📊 **Capacités:**
- API par défaut: {status_data.get('default_api', 'N/A')}
- Champs disponibles: {', '.join(status_data.get('valid_fields', []))}
- Limite par défaut: {status_data.get('default_limit', 10)}

🎯 **Statut global:** {status_data.get('overall_status', '⚠️ Inconnu')}
"""

def format_error_response(error_message: str, context: str = "") -> str:
    """Formate une réponse d'erreur pour MCP"""
    
    base_message = f"❌ Erreur: {error_message}"
    
    if context:
        base_message += f"\n\n🔍 **Contexte:** {context}"
    
    base_message += "\n\n💡 **Suggestions:**"
    base_message += "\n- Vérifiez votre fichier .env"
    base_message += "\n- Contrôlez google-credentials.json"
    base_message += "\n- Testez avec get_agent_status"
    
    return base_message