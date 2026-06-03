import json
import os
from google_auth_oauthlib.flow import InstalledAppFlow

# Cakupan akses yang dibutuhkan (Full Drive access)
SCOPES = ['https://www.googleapis.com/auth/drive']

def generate_token():
    if not os.path.exists('credentials.json'):
        print("❌ Error: File 'credentials.json' tidak ditemukan!")
        print("👉 Pastikan file yang kamu download dari Google sudah di-rename menjadi 'credentials.json' dan diletakkan di folder ini.")
        return

    # Jalankan flow login di browser
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)

    # Simpan data kredensial ke format JSON (bukan binary)
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes
    }

    with open('token_drive.json', 'w', encoding='utf-8') as f:
        json.dump(token_data, f, indent=4)
    
    print("✅ Berhasil! File 'token_drive.json' (format teks) telah dibuat. Sekarang kamu bisa melakukan git push.")

if __name__ == '__main__':
    generate_token()