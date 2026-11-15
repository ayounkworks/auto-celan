# Auto Celan — Manga Text Removal

## Local CLI

Auto Celan dapat memproses folder chapter secara langsung. Tidak perlu menyalin gambar ke folder `input/`.

```powershell
# Folder chapter langsung, default 10 workers
python main.py "C:\Komik\Chapter 0067"

# Atur jumlah workers
python main.py "C:\Komik\Chapter 0067" -w 10

# Tentukan folder output
python main.py "C:\Komik\Chapter 0067" -w 10 -o "D:\AutoCelan\Output"

# Mode kompatibilitas lama: membaca folder input/
python main.py local
```

Default output adalah `output/` di dalam repository. Setiap proses membuat subfolder unik dengan nama chapter dan timestamp.

### Perintah singkat Windows PowerShell

Agar bisa memakai perintah `ac` tanpa menulis `python main.py`:

1. Jalankan `notepad $PROFILE`.
2. Tambahkan fungsi berikut dan sesuaikan lokasi repository:
   ```powershell
   function ac { python "C:\PATH\TO\auto-celan\main.py" @args }
   ```
3. Simpan, tutup, lalu buka PowerShell baru.

Contoh penggunaan:
```powershell
ac "C:\Users\ayoun\Downloads\Chapter 0067" -w 10
ac "C:\Users\ayoun\Downloads\Chapter 0067"
```

Gunakan tanda kutip untuk folder yang memiliki spasi, tanda kurung, atau karakter khusus.

### CLI options

| Argument | Default | Fungsi |
|---|---:|---|
| `folder` | `input/` | Folder chapter sumber |
| `-w, --workers` | `10` | Jumlah gambar yang diproses bersamaan (1–64) |
| `-o, --output` | `output/` | Folder induk hasil |

Workers mengatur concurrency di proses lokal. Kapasitas aktual tetap dibatasi kapasitas endpoint/worker RunPod.

## Server Mode

```bash
python main.py
# atau
uvicorn main:app --host 0.0.0.0 --port 8080
```

Protected API endpoints memerlukan `AUTO_CELAN_API_KEY` sebagai Bearer token. Endpoint `/health` tidak memerlukan API key.

## Project Structure

- `main.py`: entry point CLI dan FastAPI
- `local_runner.py`: pemrosesan folder lokal
- `core/`: konfigurasi, pipeline, Google Drive, database, RunPod client
- `runpod/`: handler dan image inference RunPod
- `test/`: pengujian