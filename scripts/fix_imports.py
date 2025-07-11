#!/usr/bin/env python3
"""
Script pour corriger automatiquement la structure des imports Python
"""

import os
from pathlib import Path

def fix_project_structure():
    """Corrige la structure de fichiers pour permettre les imports"""
    
    print("🔧 Correction de la structure de fichiers pour les imports Python...")
    
    # Dossier racine du projet
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"📁 Projet détecté: {project_root}")
    
    # Fichiers __init__.py requis
    required_init_files = [
        project_root / "src" / "__init__.py",
        project_root / "src" / "agent" / "__init__.py", 
        project_root / "src" / "agent" / "mcp" / "__init__.py",
    ]
    
    # Dossiers à créer si nécessaire
    required_dirs = [
        project_root / "src" / "agent" / "mcp",
        project_root / "scripts",
    ]
    
    created_count = 0
    
    # Créer les dossiers
    for dir_path in required_dirs:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Dossier créé: {dir_path.relative_to(project_root)}")
                created_count += 1
            except Exception as e:
                print(f"❌ Erreur création dossier {dir_path}: {e}")
    
    # Créer les fichiers __init__.py
    for init_file in required_init_files:
        if not init_file.exists():
            try:
                # Créer le dossier parent si nécessaire
                init_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Contenu du fichier __init__.py selon le dossier
                if "agent" in str(init_file) and "mcp" not in str(init_file):
                    content = '''"""
Agent LangGraph pour API to Google Sheets
"""
'''
                elif "mcp" in str(init_file):
                    content = '''"""
Serveur MCP pour l'agent API to Sheets
"""
'''
                else:
                    content = '''"""
Package Python
"""
'''
                
                # Écrire le fichier
                init_file.write_text(content.strip())
                print(f"✅ Fichier créé: {init_file.relative_to(project_root)}")
                created_count += 1
                
            except Exception as e:
                print(f"❌ Erreur création fichier {init_file}: {e}")
        else:
            print(f"✅ Fichier existe: {init_file.relative_to(project_root)}")
    
    # Vérifier les fichiers principaux
    important_files = [
        (project_root / "src" / "agent" / "graph.py", "Agent LangGraph principal"),
        (project_root / ".env", "Variables d'environnement"),
        (project_root / "google-credentials.json", "Credentials Google Sheets"),
    ]
    
    print(f"\n🔍 Vérification des fichiers importants:")
    for file_path, description in important_files:
        if file_path.exists():
            print(f"✅ {description}: {file_path.relative_to(project_root)}")
        else:
            print(f"❌ {description}: {file_path.relative_to(project_root)} MANQUANT")
    
    # Test d'import
    print(f"\n🧪 Test des imports Python:")
    
    import sys
    sys.path.insert(0, str(project_root / "src"))
    
    try:
        import agent
        print("✅ Import 'agent' réussi")
        
        try:
            from agent import graph
            print("✅ Import 'agent.graph' réussi")
            
            # Vérifier les attributs principaux
            required_attrs = ['graph', 'DEFAULT_API_URL', 'DEFAULT_LIMIT']
            for attr in required_attrs:
                if hasattr(graph, attr):
                    print(f"✅ Attribut '{attr}' trouvé")
                else:
                    print(f"⚠️ Attribut '{attr}' manquant")
                    
        except ImportError as e:
            print(f"❌ Import 'agent.graph' échoué: {e}")
            
    except ImportError as e:
        print(f"❌ Import 'agent' échoué: {e}")
    
    print(f"\n🎯 Résumé:")
    print(f"   📁 {created_count} fichier(s)/dossier(s) créé(s)")
    if created_count > 0:
        print("   🔄 Redémarrez votre environnement Python pour appliquer les changements")
    print("   💡 Utilisez le serveur MCP mis à jour avec diagnostic intégré")
    
    return created_count

if __name__ == "__main__":
    try:
        fix_project_structure()
        print("\n✅ Correction terminée avec succès !")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        exit(1)