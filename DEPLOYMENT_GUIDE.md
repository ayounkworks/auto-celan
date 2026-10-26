# Deployment Guide — Railway

## Kenapa Railway?

Pipeline jalan di server cloud Singapore/US, bukan di laptop kamu.
Semua traffic (Drive ↔ Vision ↔ RunPod) lewat backbone datacenter.
Laptop kamu cuma kirim 1 request kecil → tunggu selesai.

---

## Langkah 1: Push ke GitHub

```bash
# Di folder project
git init -b main
git add .
git commit -m "initial"

# Jika branch masih 'master', paksa ke 'main':
git branch -M main

# Hubungkan ke repo GitHub (ganti USERNAME)
git remote add origin https://github.com/USERNAME/auto-celan.git
git push -u origin main --force
```

> ⚠️ .gitignore sudah exclude .env, token_drive.json, credentials.json
> Jangan pernah commit file-file itu.

---

## Langkah 2: Buat project di Railway

1. Buka https://railway.app → Login dengan GitHub
2. Klik **New Project** → **Deploy from GitHub repo**
3. Pilih repo `auto-celan`
4. Railway otomatis detect Dockerfile → klik **Deploy**

---

## Langkah 3: Set Environment Variables

Di Railway dashboard → project kamu → tab **Variables** → tambahkan:

| Variable | Value |
|---|---|
| `GOOGLE_VISION_API_KEY` | Opsional / kompatibilitas lama |
| `DRIVE_OUTPUT_FOLDER_ID` | ID folder output Google Drive, atau kosong untuk root Drive |
| `RUNPOD_API_KEY` | API key RunPod dari dashboard milikmu |
| `RUNPOD_ENDPOINT_ID` | ID endpoint RunPod milikmu |
| `DRIVE_TOKEN_JSON` | Token OAuth Google Drive dalam format JSON satu baris |
| `OUTPUT_DELETE_DELAY_MINUTES` | `1440` — hapus output 24 jam setelah job selesai |
| `AUTO_CELAN_API_KEY` | Buat secret acak panjang untuk proteksi endpoint API |

### Nilai `DRIVE_TOKEN_JSON`

Copy isi file `token_drive.json` kamu dalam satu baris JSON:

```
{"token": "<ACCESS_TOKEN>", "refresh_token": "<REFRESH_TOKEN>", "token_uri": "https://oauth2.googleapis.com/token", "client_id": "<OAUTH_CLIENT_ID>", "client_secret": "<OAUTH_CLIENT_SECRET>", "scopes": ["https://www.googleapis.com/auth/drive"]}
```

> Ini sudah ditangani di `core/drive.py` — env var `DRIVE_TOKEN_JSON`
> dibaca lebih prioritas dari file `token_drive.json`.
>
> Jangan menempelkan token asli ke README, issue, atau log. Endpoint API selain `/health`
> memerlukan `Authorization: Bearer <AUTO_CELAN_API_KEY>`; request API tanpa key yang
> dikonfigurasi akan ditolak.

---

## Langkah 4: Dapatkan URL

Setelah deploy selesai, Railway kasih URL seperti:
```
https://auto-celan-production.up.railway.app
```

Test:
```bash
curl https://auto-celan-production.up.railway.app/health
# → {"status": "ok"}
```

---

## Langkah 5: Kirim job dari laptop

Sekarang laptop kamu **tidak perlu internet kencang** — cukup kirim 1 POST kecil:

```bash
# Isi dengan nilai yang sama seperti AUTO_CELAN_API_KEY di Railway.
export AUTO_CELAN_API_KEY="<API_KEY_DARI_RAILWAY>"

# Mulai proses folder Drive
curl -X POST https://auto-celan-production.up.railway.app/clean \
  -H "Authorization: Bearer $AUTO_CELAN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"folder_url": "https://drive.google.com/drive/folders/FOLDER_ID_KAMU"}'

# Response:
# {"job_id": "abc12345", "message": "Pipeline started"}
```

```bash
# Cek status
curl https://auto-celan-production.up.railway.app/job/abc12345 \
  -H "Authorization: Bearer $AUTO_CELAN_API_KEY"
```

---

## Cara lihat log real-time di Railway

Railway dashboard → project → **Deployments** → klik deploy aktif → **View Logs**

Atau pakai Railway CLI:
```bash
npm install -g @railway/cli
railway login
railway logs
```

---

## Estimasi biaya Railway

| Plan | CPU | RAM | Biaya |
|------|-----|-----|-------|
| Hobby (gratis) | shared | 512MB | $0 — ada sleep setelah idle |
| Starter ($5/bln) | shared | 512MB | $5 — tidak sleep, recommended |
| Pro ($20/bln) | dedicated | 8GB | $20 — untuk volume tinggi |

**Untuk auto_celan: Starter ($5/bln) sudah lebih dari cukup.**
Bottleneck ada di RunPod (GPU), bukan di server ini.

---

## Update kode

Setiap `git push` ke branch `main` → Railway otomatis redeploy.
Tidak perlu rebuild Docker RunPod selama kamu tidak ubah `runpod/`.

```bash
# Edit kode → push → Railway auto-deploy dalam ~2 menit
git add .
git commit -m "fix: ..."
git push origin main
```

---

## Perbandingan kecepatan (estimasi)

| Mode | Download 50 hal | Upload hasil | Total 50 file |
|------|----------------|--------------|---------------|
| Laptop Indonesia | ~15 menit | ~10 menit | ~60 menit |
| Railway (server) | ~1 menit | ~1 menit | ~10 menit |

Perbedaan utama: traffic Drive ↔ Vision ↔ RunPod jalan di backbone
datacenter, bukan lewat ISP Indonesia.
