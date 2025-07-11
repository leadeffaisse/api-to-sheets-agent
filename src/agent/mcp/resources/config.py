"""
Ressources de configuration MCP
"""

import json
import os
from typing import List
from mcp.types import Resource

class ConfigResources:
    def __init__(self, server):
        self.server = server
        # Note: On n'enregistre pas les handlers ici pour éviter les conflits
    
    @staticmethod
    def get_resources() -> List[Resource]:
        """Retourne la liste des ressources disponibles"""
        return [
            Resource(
                uri="config://agent-config",
                name="Configuration de l'agent",
                description="Configuration actuelle de l'agent API to Sheets",
                mimeType="application/json"
            ),
            Resource(
                uri="config://api-fields",
                name="Champs API disponibles",
                description="Liste des champs disponibles pour les requêtes API",
                mimeType="application/json"
            ),
            Resource(
                uri="state://current-state",
                name="État actuel",
                description="État actuel de l'agent (dernière exécution)",
                mimeType="application/json"
            )
        ]
    
    @staticmethod
    def read_resource(uri: str) -> str:
        """Lit le contenu d'une ressource"""
        try:
            # Import dynamique pour éviter les erreurs circulaires
            from agent.graph import (
                DEFAULT_API_URL, DEFAULT_LIMIT, VALID_API_FIELDS,
                OPENAI_MODEL, OPENAI_TEMPERATURE
            )
            
            if uri == "config://agent-config":
                config = {
                    "default_api_url": DEFAULT_API_URL,
                    "default_limit": DEFAULT_LIMIT,
                    "valid_fields": VALID_API_FIELDS,
                    "model": OPENAI_MODEL,
                    "temperature": OPENAI_TEMPERATURE,
                    "environment_variables": {
                        "OPENAI_API_KEY": "✅" if os.getenv("OPENAI_API_KEY") else "❌",
                        "GOOGLE_CREDENTIALS_PATH": "✅" if os.path.exists(os.getenv("GOOGLE_CREDENTIALS_PATH", "")) else "❌",
                        "LANGSMITH_API_KEY": "✅" if os.getenv("LANGSMITH_API_KEY") else "❌"
                    }
                }
                return json.dumps(config, indent=2)
            
            elif uri == "config://api-fields":
                fields_info = {
                    "valid_fields": VALID_API_FIELDS,
                    "field_descriptions": {
                        "userId": "ID de l'utilisateur",
                        "id": "ID unique du post",
                        "title": "Titre du post",
                        "body": "Contenu du post"
                    },
                    "usage_examples": [
                        "récupère 5 posts avec title et id",
                        "obtiens 10 posts avec tous les champs",
                        "prends 3 posts avec seulement le title"
                    ]
                }
                return json.dumps(fields_info, indent=2)
            
            elif uri == "state://current-state":
                from agent.graph import get_initial_state
                initial_state = get_initial_state()
                # Convertir en dict sérialisable
                serializable_state = {}
                for key, value in initial_state.items():
                    try:
                        json.dumps(value)  # Test de sérialisation
                        serializable_state[key] = value
                    except:
                        serializable_state[key] = str(value)
                
                return json.dumps(serializable_state, indent=2, default=str)
            
            else:
                raise ValueError(f"Ressource inconnue: {uri}")
                
        except Exception as e:
            error_response = {
                "error": f"Erreur lors de la lecture de la ressource: {str(e)}",
                "uri": uri
            }
            return json.dumps(error_response, indent=2)