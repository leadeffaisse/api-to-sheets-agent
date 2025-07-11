#!/usr/bin/env python3
"""
Configuration Claude Desktop corrigée
"""

import os
import json
import platform
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def get_claude_config_path():
    """Retourne le chemin du fichier de config Claude Desktop"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    elif system == "Windows":
        return Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json"
    elif system == "Linux":
        return Path.home() / ".config/claude/claude_desktop_config.json"
    else:
        raise OSError(f"Système non supporté: {system}")

def setup_claude_config():
    """Configure Claude Desktop avec les bons chemins et variables"""
    
    print("🔧 Configuration de Claude Desktop pour MCP...")
    
    # Chemin vers le serveur MCP
    project_root = Path(__file__).parent.parent.absolute()
    server_path = project_root / "src" / "agent" / "mcp" / "server.py"
    
    if not server_path.exists():
        print(f"❌ Serveur MCP introuvable: {server_path}")
        return False
    
    # Chemin vers le Python du venv
    venv_python = project_root / "venv" / "Scripts" / "python.exe"  # Windows
    if not venv_python.exists():
        venv_python = project_root / "venv" / "bin" / "python"  # Linux/Mac
    
    # Utiliser le Python du venv s'il existe, sinon Python système
    if venv_python.exists():
        python_cmd = str(venv_python)
        print(f"✅ Utilisation du Python du venv: {python_cmd}")
    else:
        python_cmd = "python"
        print("⚠️ Python du venv non trouvé, utilisation du Python système")
    
    # Récupérer la clé OpenAI depuis .env
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ OPENAI_API_KEY non trouvée dans .env")
        openai_key = input("Entrez votre clé OpenAI: ").strip()
    else:
        print("✅ Clé OpenAI trouvée dans .env")
    
    # Configuration MCP
    mcp_config = {
        "api-sheets-agent": {
            "command": python_cmd,
            "args": [str(server_path)],
            "env": {
                "PYTHONPATH": str(project_root),
                "GOOGLE_CREDENTIALS_PATH": str(project_root / "google-credentials.json"),
                "GOOGLE_PERSONAL_EMAIL": os.getenv("GOOGLE_PERSONAL_EMAIL", ""),
                "OPENAI_API_KEY": openai_key,
                "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY", ""),
            }
        }
    }
    
    # Chemin de config Claude Desktop
    config_path = get_claude_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Lire la configuration existante
    if config_path.exists():
        with open(config_path, 'r') as f:
            try:
                existing_config = json.load(f)
            except json.JSONDecodeError:
                existing_config = {}
    else:
        existing_config = {}
    
    # Ajouter la configuration MCP
    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}
    
    existing_config["mcpServers"].update(mcp_config)
    
    # Sauvegarder
    with open(config_path, 'w') as f:
        json.dump(existing_config, f, indent=2)
    
    print(f"✅ Configuration sauvegardée: {config_path}")
    print("🔄 Redémarrez Claude Desktop pour appliquer les changements")
    
    # Afficher la configuration pour vérification
    print("\n📋 Configuration MCP:")
    print(f"  Python: {python_cmd}")
    print(f"  Serveur: {server_path}")
    print(f"  OpenAI Key: {'✅ Configurée' if openai_key else '❌ Manquante'}")
    
    return True

def main():
    """Point d'entrée principal"""
    try:
        success = setup_claude_config()
        if success:
            print("\n🎯 Configuration terminée avec succès !")
            print("\n📝 Étapes suivantes :")
            print("  1. Redémarrer Claude Desktop complètement")
            print("  2. Tester avec la commande 'ping'")
        else:
            print("\n❌ Échec de la configuration")
            return 1
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())