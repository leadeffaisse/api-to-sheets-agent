#!/usr/bin/env python3
"""
Script pour nettoyer le Google Drive d'un compte de service
"""

import os
from pathlib import Path
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta

def setup_drive_service():
    """Configure le service Google Drive"""
    # Chemin vers les credentials
    project_root = Path(__file__).parent
    credentials_path = project_root / "google-credentials.json"
    
    if not credentials_path.exists():
        print(f"❌ Fichier credentials non trouvé: {credentials_path}")
        return None
    
    # Scopes nécessaires
    scopes = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file'
    ]
    
    try:
        creds = Credentials.from_service_account_file(
            str(credentials_path), 
            scopes=scopes
        )
        service = build('drive', 'v3', credentials=creds)
        print("✅ Service Google Drive configuré")
        return service
    except Exception as e:
        print(f"❌ Erreur configuration Drive: {e}")
        return None

def list_all_files(service, max_files=100):
    """Liste tous les fichiers du Drive"""
    try:
        print(f"\n📋 Listing des fichiers (max {max_files})...")
        
        results = service.files().list(
            pageSize=max_files,
            fields="nextPageToken, files(id, name, createdTime, size, mimeType, trashed)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print("📂 Aucun fichier trouvé")
            return []
        
        print(f"📁 {len(files)} fichier(s) trouvé(s):")
        
        total_size = 0
        for i, file in enumerate(files, 1):
            name = file['name']
            file_id = file['id']
            created = file.get('createdTime', 'Inconnue')
            size = int(file.get('size', 0))
            mime_type = file.get('mimeType', 'Inconnu')
            trashed = file.get('trashed', False)
            
            total_size += size
            
            status = "🗑️" if trashed else "📄"
            size_mb = size / (1024 * 1024) if size > 0 else 0
            
            print(f"  {status} {i:2d}. {name[:50]:<50} | {size_mb:6.2f} MB | {created[:10]}")
            print(f"       ID: {file_id}")
            print(f"       Type: {mime_type}")
            print()
        
        total_mb = total_size / (1024 * 1024)
        print(f"💾 Taille totale: {total_mb:.2f} MB")
        
        return files
        
    except Exception as e:
        print(f"❌ Erreur listing: {e}")
        return []

def delete_files_by_pattern(service, pattern="", confirm=True):
    """Supprime les fichiers contenant un pattern dans le nom"""
    try:
        print(f"\n🔍 Recherche fichiers contenant: '{pattern}'")
        
        # Construire la requête
        query = f"name contains '{pattern}'" if pattern else "trashed=false"
        
        results = service.files().list(
            q=query,
            fields="files(id, name, createdTime, size)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print(f"📂 Aucun fichier trouvé avec le pattern '{pattern}'")
            return 0
        
        print(f"🎯 {len(files)} fichier(s) trouvé(s) à supprimer:")
        
        total_size = 0
        for file in files:
            name = file['name']
            size = int(file.get('size', 0))
            total_size += size
            size_mb = size / (1024 * 1024) if size > 0 else 0
            created = file.get('createdTime', 'Inconnue')
            
            print(f"  📄 {name[:60]:<60} | {size_mb:6.2f} MB | {created[:10]}")
        
        total_mb = total_size / (1024 * 1024)
        print(f"\n💾 Taille totale à libérer: {total_mb:.2f} MB")
        
        if confirm:
            response = input(f"\n⚠️  Confirmer la suppression de {len(files)} fichier(s) ? (oui/non): ")
            if response.lower() not in ['oui', 'o', 'yes', 'y']:
                print("❌ Suppression annulée")
                return 0
        
        # Supprimer les fichiers
        deleted_count = 0
        for file in files:
            try:
                service.files().delete(fileId=file['id']).execute()
                print(f"✅ Supprimé: {file['name']}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Erreur suppression {file['name']}: {e}")
        
        print(f"\n🎉 {deleted_count}/{len(files)} fichier(s) supprimé(s)")
        print(f"💾 Espace libéré: ~{total_mb:.2f} MB")
        
        return deleted_count
        
    except Exception as e:
        print(f"❌ Erreur suppression: {e}")
        return 0

def delete_old_files(service, days_old=7, confirm=True):
    """Supprime les fichiers plus anciens que X jours"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cutoff_str = cutoff_date.isoformat() + 'Z'
        
        print(f"\n🗓️  Recherche fichiers créés avant: {cutoff_date.strftime('%Y-%m-%d %H:%M')}")
        
        query = f"createdTime < '{cutoff_str}'"
        
        results = service.files().list(
            q=query,
            fields="files(id, name, createdTime, size)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print(f"📂 Aucun fichier trouvé plus ancien que {days_old} jours")
            return 0
        
        print(f"🎯 {len(files)} fichier(s) ancien(s) trouvé(s):")
        
        total_size = 0
        for file in files:
            name = file['name']
            size = int(file.get('size', 0))
            total_size += size
            size_mb = size / (1024 * 1024) if size > 0 else 0
            created = file.get('createdTime', 'Inconnue')
            
            print(f"  📄 {name[:60]:<60} | {size_mb:6.2f} MB | {created[:10]}")
        
        total_mb = total_size / (1024 * 1024)
        print(f"\n💾 Taille totale à libérer: {total_mb:.2f} MB")
        
        if confirm:
            response = input(f"\n⚠️  Confirmer la suppression des fichiers de plus de {days_old} jours ? (oui/non): ")
            if response.lower() not in ['oui', 'o', 'yes', 'y']:
                print("❌ Suppression annulée")
                return 0
        
        # Supprimer les fichiers
        deleted_count = 0
        for file in files:
            try:
                service.files().delete(fileId=file['id']).execute()
                print(f"✅ Supprimé: {file['name']}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Erreur suppression {file['name']}: {e}")
        
        print(f"\n🎉 {deleted_count}/{len(files)} fichier(s) supprimé(s)")
        print(f"💾 Espace libéré: ~{total_mb:.2f} MB")
        
        return deleted_count
        
    except Exception as e:
        print(f"❌ Erreur suppression: {e}")
        return 0

def empty_trash(service):
    """Vide la corbeille"""
    try:
        print("\n🗑️  Vidage de la corbeille...")
        service.files().emptyTrash().execute()
        print("✅ Corbeille vidée")
        return True
    except Exception as e:
        print(f"❌ Erreur vidage corbeille: {e}")
        return False

def get_drive_usage(service):
    """Affiche l'utilisation du Drive"""
    try:
        print("\n📊 Analyse de l'utilisation du Drive...")
        
        about = service.about().get(fields="storageQuota").execute()
        quota = about.get('storageQuota', {})
        
        limit = int(quota.get('limit', 0))
        usage = int(quota.get('usage', 0))
        usage_in_drive = int(quota.get('usageInDrive', 0))
        
        if limit > 0:
            limit_gb = limit / (1024**3)
            usage_gb = usage / (1024**3)
            drive_gb = usage_in_drive / (1024**3)
            
            percentage = (usage / limit) * 100
            
            print(f"💾 Utilisation totale: {usage_gb:.2f} GB / {limit_gb:.2f} GB ({percentage:.1f}%)")
            print(f"📁 Utilisation Drive: {drive_gb:.2f} GB")
            print(f"🔴 Espace libre: {(limit_gb - usage_gb):.2f} GB")
            
            if percentage > 90:
                print("⚠️  ATTENTION: Quota presque plein!")
            elif percentage > 80:
                print("⚠️  Avertissement: Quota à plus de 80%")
        else:
            print("❓ Impossible de récupérer les informations de quota")
            
    except Exception as e:
        print(f"❌ Erreur récupération usage: {e}")

def main():
    """Fonction principale"""
    print("🧹 === NETTOYAGE GOOGLE DRIVE COMPTE DE SERVICE ===")
    
    # Configurer le service
    service = setup_drive_service()
    if not service:
        return
    
    # Afficher l'utilisation
    get_drive_usage(service)
    
    while True:
        print("\n" + "="*60)
        print("🛠️  MENU NETTOYAGE")
        print("="*60)
        print("1. 📋 Lister tous les fichiers")
        print("2. 🔍 Supprimer par nom/pattern")
        print("3. 🗓️  Supprimer fichiers anciens")
        print("4. 🗑️  Vider la corbeille")
        print("5. 📊 Afficher utilisation")
        print("6. 🚀 Nettoyage rapide (fichiers MCP)")
        print("0. ❌ Quitter")
        print("="*60)
        
        choice = input("Votre choix: ").strip()
        
        if choice == "1":
            list_all_files(service)
            
        elif choice == "2":
            pattern = input("Pattern à rechercher (ex: 'MCP', 'Test', 'Posts'): ").strip()
            if pattern:
                delete_files_by_pattern(service, pattern)
            
        elif choice == "3":
            try:
                days = int(input("Supprimer fichiers plus anciens que (jours): "))
                delete_old_files(service, days)
            except ValueError:
                print("❌ Nombre de jours invalide")
            
        elif choice == "4":
            empty_trash(service)
            
        elif choice == "5":
            get_drive_usage(service)
            
        elif choice == "6":
            print("🚀 Nettoyage rapide des fichiers de test MCP...")
            deleted = 0
            deleted += delete_files_by_pattern(service, "MCP", confirm=False)
            deleted += delete_files_by_pattern(service, "Test", confirm=False)
            deleted += delete_files_by_pattern(service, "Posts", confirm=False)
            deleted += delete_files_by_pattern(service, "API_Data", confirm=False)
            empty_trash(service)
            print(f"🎉 Nettoyage terminé! {deleted} fichier(s) supprimé(s)")
            
        elif choice == "0":
            print("👋 Au revoir!")
            break
            
        else:
            print("❌ Choix invalide")

if __name__ == "__main__":
    main()