"""
Outils MCP niveau utilisateur final
Interface simple pour les cas d'usage courants
"""

import asyncio
import json
import logging
from typing import List
from mcp.types import Tool, TextContent

logger = logging.getLogger("mcp.tools.basic")

class BasicTools:
    def __init__(self, server):
        self.server = server
        # Note: On n'enregistre pas les handlers ici pour éviter les conflits
        # Les handlers sont définis directement dans server.py
    
    @staticmethod
    def get_tools() -> List[Tool]:
        """Retourne la liste des outils basiques"""
        return [
            Tool(
                name="fetch_api_to_sheets",
                description="Récupère des données d'une API et les exporte vers Google Sheets",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Requête en langage naturel décrivant les données à récupérer"
                        },
                        "api_url": {
                            "type": "string",
                            "description": "URL de l'API à interroger (optionnel)",
                            "default": "https://jsonplaceholder.typicode.com/posts"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="validate_api_query",
                description="Valide et analyse une requête API sans l'exécuter",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Requête à valider"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_agent_status",
                description="Obtient le statut actuel de l'agent et ses capacités",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]