#!/usr/bin/env python3
"""
Script de nettoyage des Google Sheets créés par l'agent
Usage: python cleanup_sheets.py
"""

import os
import sys
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# Configuration - utilisez les mêmes variables que votre projet
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "./google-credentials.json")

def setup_drive_service():
    """Configuration du service Google Drive"""
    try:
        print("🔧 Configuration du service Google Drive...")
        
        # Vérifier que le fichier credentials existe
        if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
            print(f"❌ Fichier credentials non trouvé: {GOOGLE_CREDENTIALS_PATH}")
            return None
        
        # Définir les scopes
        scopes = [
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Charger les credentials
        creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=scopes)
        
        # Créer le service Drive
        drive_service = build('drive', 'v3', credentials=creds)
        
        print("✅ Service Google Drive configuré avec succès")
        return drive_service
        
    except Exception as e:
        print(f"❌ Erreur configuration Drive: {e}")
        return None

def list_api_data_sheets(drive_service):
    """Liste tous les sheets créés par l'API"""
    try:
        print("🔍 Recherche des Google Sheets API_Data...")
        
        # Chercher tous les fichiers Google Sheets avec le préfixe API_Data
        query = "name contains 'API_Data_' and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
        
        results = drive_service.files().list(
            q=query,
            fields="files(id, name, createdTime, webViewLink, parents)",
            orderBy="createdTime desc",
            pageSize=50
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print("✅ Aucun sheet API_Data trouvé.")
            return []
        
        print(f"📊 {len(files)} sheets trouvés:")
        print("-" * 80)
        
        for i, file in enumerate(files, 1):
            try:
                created_time = datetime.fromisoformat(file['createdTime'].replace('Z', '+00:00'))
                created_str = created_time.strftime('%Y-%m-%d %H:%M:%S')
            except:
                created_str = "Date inconnue"
            
            print(f"{i:2d}. {file['name']}")
            print(f"    ID: {file['id']}")
            print(f"    Créé: {created_str}")
            print(f"    URL: {file.get('webViewLink', 'Non disponible')}")
            print()
        
        return files
        
    except Exception as e:
        print(f"❌ Erreur lors de la recherche: {e}")
        return []

def delete_all_api_sheets(drive_service, files, confirm=True):
    """Supprime tous les sheets API_Data"""
    if not files:
        print("ℹ️  Aucun sheet à supprimer.")
        return
    
    print(f"\n⚠️  ATTENTION: Vous allez supprimer {len(files)} sheets!")
    
    if confirm:
        print("\nListe des sheets qui seront supprimés:")
        for i, file in enumerate(files, 1):
            print(f"  {i}. {file['name']}")
        
        print(f"\n🗑️  Confirmer la suppression de {len(files)} sheets ?")
        response = input("Tapez 'OUI' en majuscules pour confirmer: ").strip()
        
        if response != 'OUI':
            print("❌ Suppression annulée.")
            return
    
    print(f"\n🗑️  Suppression en cours...")
    deleted_count = 0
    errors = []
    
    for i, file in enumerate(files, 1):
        try:
            print(f"[{i}/{len(files)}] Suppression de: {file['name']}")
            drive_service.files().delete(fileId=file['id']).execute()
            print(f"    ✅ Supprimé avec succès")
            deleted_count += 1
        except Exception as e:
            error_msg = f"    ❌ Erreur: {e}"
            print(error_msg)
            errors.append(f"{file['name']}: {e}")
    
    # Résumé
    print(f"\n" + "="*60)
    print(f"🎯 RÉSUMÉ DE LA SUPPRESSION")
    print(f"="*60)
    print(f"✅ Sheets supprimés: {deleted_count}/{len(files)}")
    
    if errors:
        print(f"❌ Erreurs: {len(errors)}")
        print("\nDétails des erreurs:")
        for error in errors[:5]:  # Montrer max 5 erreurs
            print(f"  • {error}")
        if len(errors) > 5:
            print(f"  ... et {len(errors) - 5} autres erreurs")
    
    print(f"\n🎉 Nettoyage terminé!")

def delete_sheets_older_than(drive_service, files, days):
    """Supprime les sheets plus anciens que X jours"""
    try:
        cutoff_date = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=days)
        old_files = []
        
        for file in files:
            try:
                created_time = datetime.fromisoformat(file['createdTime'].replace('Z', '+00:00'))
                if created_time < cutoff_date:
                    old_files.append(file)
            except:
                continue
        
        if not old_files:
            print(f"✅ Aucun sheet de plus de {days} jours trouvé.")
            return
        
        print(f"\n📅 {len(old_files)} sheets de plus de {days} jours trouvés:")
        for file in old_files:
            try:
                created_time = datetime.fromisoformat(file['createdTime'].replace('Z', '+00:00'))
                created_str = created_time.strftime('%Y-%m-%d')
            except:
                created_str = "Date inconnue"
            print(f"  • {file['name']} (créé le {created_str})")
        
        # Supprimer les anciens sheets
        delete_all_api_sheets(drive_service, old_files, confirm=True)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

def delete_specific_sheets(drive_service, sheet_ids):
    """Supprime des sheets spécifiques par leurs IDs"""
    if not sheet_ids:
        print("ℹ️  Aucun ID fourni.")
        return
    
    print(f"🗑️  Suppression de {len(sheet_ids)} sheets spécifiques...")
    
    deleted_count = 0
    for i, sheet_id in enumerate(sheet_ids, 1):
        try:
            # Récupérer le nom avant suppression
            file_info = drive_service.files().get(fileId=sheet_id, fields='name').execute()
            file_name = file_info.get('name', 'Nom inconnu')
            
            print(f"[{i}/{len(sheet_ids)}] Suppression: {file_name}")
            drive_service.files().delete(fileId=sheet_id).execute()
            print(f"    ✅ Supprimé avec succès")
            deleted_count += 1
            
        except Exception as e:
            print(f"    ❌ Erreur: {e}")
    
    print(f"\n🎯 {deleted_count}/{len(sheet_ids)} sheets supprimés.")

def main_menu():
    """Menu principal du script de nettoyage"""
    print("\n" + "="*60)
    print("🧹 NETTOYAGE DES GOOGLE SHEETS API_DATA")
    print("="*60)
    
    # Test de la configuration
    drive_service = setup_drive_service()
    if not drive_service:
        print("❌ Impossible de continuer sans service Google Drive.")
        return
    
    while True:
        print(f"\n📋 OPTIONS DISPONIBLES:")
        print("1. 📊 Lister tous les sheets API_Data")
        print("2. 🗑️  Supprimer TOUS les sheets API_Data")
        print("3. 📅 Supprimer les sheets anciens (> X jours)")
        print("4. 🎯 Supprimer des sheets spécifiques (par ID)")
        print("5. 🚪 Quitter")
        print()
        
        choice = input("Choisissez une option (1-5): ").strip()
        
        if choice == "1":
            # Lister les sheets
            files = list_api_data_sheets(drive_service)
            
        elif choice == "2":
            # Supprimer tous les sheets
            files = list_api_data_sheets(drive_service)
            if files:
                delete_all_api_sheets(drive_service, files)
            
        elif choice == "3":
            # Supprimer les anciens sheets
            try:
                days = int(input("\nSupprimer les sheets de plus de combien de jours ? "))
                files = list_api_data_sheets(drive_service)
                if files:
                    delete_sheets_older_than(drive_service, files, days)
            except ValueError:
                print("❌ Veuillez entrer un nombre valide.")
                
        elif choice == "4":
            # Supprimer des sheets spécifiques
            print("\n📝 Entrez les IDs des sheets à supprimer:")
            print("   (séparés par des virgules)")
            print("   Exemple: 1ABC...,2DEF...,3GHI...")
            ids_input = input("\nIDs: ").strip()
            
            if ids_input:
                sheet_ids = [id.strip() for id in ids_input.split(',') if id.strip()]
                delete_specific_sheets(drive_service, sheet_ids)
            else:
                print("❌ Aucun ID fourni.")
                
        elif choice == "5":
            print("\n👋 Au revoir !")
            break
            
        else:
            print("❌ Option invalide. Choisissez entre 1 et 5.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption détectée. Au revoir !")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")

# IDs connus extraits de vos logs précédents (pour référence)
KNOWN_SHEET_IDS = [
    "1RdwO5pzJAOwyO_NTOXQqRLUrEauDe9_UXyHiuEq8gFo",
    "17gA3MCzvX4eJKZo5f1zAcKBIuVzW62TqKLKLm0VIHaU", 
    "1PcummNXacp_RZuAwn7QkN8oh8POrw3D7YKi0T3CkgsM",
    "16SJrgOEkFScJQ2CkSLkOcnNzsvsk_VcR3yaG20N4z-E",
    "1McJ36pr9w-1trr9Fx78TCTHOko1F4u9nHkrHI-AW6KA",
    "1s7cM6BEAxJ0rsExP43Hy34VZDV_qCTZBubswTEwFtnQ"
]