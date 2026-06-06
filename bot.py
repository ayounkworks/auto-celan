# ============================================================
# bot.py — Discord Bot untuk auto_celan
# ============================================================

import asyncio
import os
import sys
import socket
import uuid
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import discord
from discord import app_commands
import aiohttp
from google.cloud import vision
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────

DISCORD_TOKEN    = os.getenv("DISCORD_BOT_TOKEN")
ERROR_CHANNEL_ID = int(os.getenv("ERROR_CHANNEL_ID", "0"))

if not DISCORD_TOKEN:
    print("❌ DISCORD_BOT_TOKEN tidak ditemukan di .env")
    sys.exit(1)

# ── Import core ───────────────────────────────────────────

from core.config   import GOOGLE_API_KEY, OUTPUT_AUTO_DELETE_MINUTES
from core.database import (
    init_db, db_create_job, db_get_job,
    db_register_user, db_update_last_job,
)
from core.drive import extract_folder_id, _get_drive
import core.runpod_client as runpod_module
import core.pipeline      as pipeline_module
from core.pipeline import jobs, cancelled_jobs, run_pipeline, warmup, deletion_loop


# ── Embed builder ─────────────────────────────────────────

def _make_job_embed(job_id: str, data: dict) -> discord.Embed:
    status = data.get("status", "queued")

    color_map = {
        "queued":    discord.Color.yellow(),
        "running":   discord.Color.blue(),
        "completed": discord.Color.green(),
        "failed":    discord.Color.red(),
        "cancelled": discord.Color.dark_gray(),  # FIX: grayed() tidak ada
    }
    icon_map = {
        "queued":    "⏳",
        "running":   "⚙️",
        "completed": "✅",
        "failed":    "❌",
        "cancelled": "🚫",
    }

    embed = discord.Embed(
        title=f"{icon_map.get(status,'❓')} Job `{job_id}` — {status.upper()}",
        color=color_map.get(status, discord.Color.blurple()),
    )

    total     = data.get("total_files")     or 0
    completed = data.get("completed_files") or 0
    output_id = data.get("output_folder_id")
    result    = data.get("result_folder")
    eta       = data.get("eta")
    progress  = data.get("progress", "")
    failed    = data.get("failed_files")

    if total > 0:
        pct = int(completed / total * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        embed.add_field(
            name="Progress",
            value=f"`[{bar}] {pct}%` — {completed}/{total} file",
            inline=False,
        )

    if eta and status == "running":
        embed.add_field(name="ETA", value=eta, inline=True)

    if progress and status == "running":
        short = progress[:100] + "..." if len(progress) > 100 else progress
        embed.add_field(name="Last update", value=f"`{short}`", inline=False)

    if status == "completed" and output_id:
        url = f"https://drive.google.com/drive/folders/{output_id}"
        # Hitung waktu penghapusan untuk timestamp Discord
        unix_ts = int(datetime.now().timestamp() + (OUTPUT_AUTO_DELETE_MINUTES * 60))
        embed.add_field(
            name="📁 Output Drive", 
            value=f"[Buka folder]({url})\n⏰ Dihapus otomatis <t:{unix_ts}:R>", 
            inline=False
        )

    if failed and failed != "[]":
        failed_list = json.loads(failed) if isinstance(failed, str) else failed
        if failed_list:
            preview = ", ".join(failed_list[:5])
            if len(failed_list) > 5:
                preview += f" (+{len(failed_list)-5} lagi)"
            embed.add_field(name="❌ Gagal", value=preview, inline=False)

    return embed


async def _notify_done(channel, job_id: str, mention: str):
    row = db_get_job(job_id)
    if not row:
        return
    data   = dict(row)
    status = data.get("status", "?")
    icon   = "✅" if status == "completed" else "❌"
    await channel.send(
        content=f"{mention} {icon} Job `{job_id}` selesai — **{status.upper()}**!",
        embed=_make_job_embed(job_id, data),
    )


# ── Bot setup ─────────────────────────────────────────────

intents = discord.Intents.default()
bot     = discord.Client(intents=intents)
tree    = app_commands.CommandTree(bot)

# FIX Bug 3: Dedicated thread pool untuk pipeline — tidak berebut dengan Discord heartbeat
_pipeline_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")

# FIX Bug 4: Semaphore global di level bot — max 2 job paralel di seluruh bot
# Sebelumnya tiap /clean buat semaphorenya sendiri → tidak ada limit antar-job
_global_job_sem: asyncio.Semaphore = None  # diinit di on_ready atau saat pertama dipakai


@bot.event
async def on_ready():
    global _global_job_sem
    await tree.sync()
    print(f"✅ Bot online: {bot.user} (ID: {bot.user.id})")

    init_db()

    # FIX Bug 4: Init global semaphore di event loop yang benar
    if _global_job_sem is None:
        _global_job_sem = asyncio.Semaphore(2)

    # FIX Bug 1: Tutup session lama sebelum buat yang baru (cegah leak saat reconnect)
    if pipeline_module._http_session and not pipeline_module._http_session.closed:
        await pipeline_module._http_session.close()

    timeout   = aiohttp.ClientTimeout(total=300, connect=20, sock_read=300)
    connector = aiohttp.TCPConnector(limit=30, ttl_dns_cache=300, family=socket.AF_INET)
    pipeline_module._http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    pipeline_module._warmup_lock  = asyncio.Lock()
    pipeline_module.vision_client = vision.ImageAnnotatorClient(
        client_options={"api_key": GOOGLE_API_KEY}
    )

    asyncio.create_task(deletion_loop())
    asyncio.create_task(warmup())


@bot.event
async def on_close():
    if pipeline_module._http_session and not pipeline_module._http_session.closed:
        await pipeline_module._http_session.close()
    _pipeline_executor.shutdown(wait=False)


# ── /help ─────────────────────────────────────────────────

@tree.command(name="help", description="Daftar semua command auto_celan")
async def cmd_help(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🤖 Auto Celan — Manga Text Removal",
        description="Bot untuk menghapus teks dari halaman manga secara otomatis.",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="`/clean <folder_url>`",
        value="Mulai proses folder Google Drive.\n"
              "Contoh: `/clean https://drive.google.com/drive/folders/xxx`",
        inline=False,
    )
    embed.add_field(name="`/status <job_id>`", value="Cek status job.", inline=False)
    embed.add_field(name="`/cancel <job_id>`", value="Batalkan job yang sedang berjalan.", inline=False)
    embed.set_footer(text="Bot akan mention kamu otomatis saat job selesai.")
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ── /clean ────────────────────────────────────────────────

@tree.command(name="clean", description="Proses folder Google Drive — hapus teks manga")
@app_commands.describe(folder_url="Link Google Drive folder berisi gambar manga")
async def cmd_clean(interaction: discord.Interaction, folder_url: str):
    await interaction.response.defer(thinking=True)

    user       = interaction.user
    channel    = interaction.channel
    discord_id = str(user.id)

    db_register_user(discord_id, str(user))
    db_update_last_job(discord_id)

    # Validasi folder
    folder_id = extract_folder_id(folder_url)
    if not folder_id or len(folder_id) < 10:
        await interaction.followup.send(
            "❌ Link Google Drive tidak valid.\n"
            "Format: `https://drive.google.com/drive/folders/FOLDER_ID`",
            ephemeral=True,
        )
        return

    # Cek folder bisa diakses + ambil nama
    try:
        meta = await asyncio.to_thread(
            lambda: _get_drive().files().get(fileId=folder_id, fields="id,name").execute()
        )
    except Exception as e:
        await interaction.followup.send(
            f"❌ Folder tidak bisa diakses: `{e}`\n"
            "Pastikan folder sudah di-share ke akun Drive bot.",
            ephemeral=True,
        )
        return

    folder_name = meta.get("name") or folder_id

    # Buat job
    job_id = str(uuid.uuid4())[:8]
    db_create_job(job_id, discord_id, folder_url, 0)
    jobs[job_id] = {
        "status":           "queued",
        "progress":         "Starting...",
        "result_folder":    None,
        "output_folder_id": None,
        "queue_position":   0,
        "total_files":      0,
        "completed_files":  0,
        "failed_files":     [],
        "eta":              "...",
        "log":              [],
    }

    # Kirim embed awal
    embed = discord.Embed(
        title=f"⏳ Job `{job_id}` dimulai",
        description=f"Folder: **{folder_name}**",
        color=discord.Color.yellow(),
    )
    embed.add_field(name="Status", value="Sedang antri / mulai...", inline=False)
    embed.set_footer(text="Embed ini akan update otomatis. Kamu akan di-mention saat selesai.")
    await interaction.followup.send(embed=embed)

    # ── Background: pipeline + live update ───────────────
    async def _live_update():
        """Update embed tiap 5 detik — jalan sebagai task terpisah."""
        try:
            msg = await interaction.original_response()
        except Exception:
            return

        while True:
            await asyncio.sleep(5)
            fresh = jobs.get(job_id) or {}
            if not fresh:
                row = db_get_job(job_id)
                if row:
                    fresh = dict(row)
            status = fresh.get("status", "queued")
            try:
                await msg.edit(embed=_make_job_embed(job_id, fresh))
            except Exception:
                pass
            if status in ("completed", "failed", "cancelled"):
                break

    async def _run_pipeline():
        """
        Jalankan pipeline di dedicated thread executor (FIX Bug 3) agar thread pool
        Discord tidak tersaturasi → heartbeat tetap jalan → tidak disconnect.
        """
        loop = asyncio.get_running_loop()

        # Buat event loop baru di thread terpisah untuk pipeline async
        def _run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                import aiohttp, socket as _socket

                async def _inner():
                    # Buat semua Semaphore di event loop ini agar tidak cross-loop error
                    import core.runpod_client as rp
                    import core.pipeline      as pl
                    rp.runpod_sem   = asyncio.Semaphore(10)
                    pl.pipeline_sem = asyncio.Semaphore(10)
                    pl.vision_sem   = asyncio.Semaphore(5)
                    pl._warmup_lock = asyncio.Lock()

                    connector = aiohttp.TCPConnector(limit=30, ttl_dns_cache=300, family=_socket.AF_INET)
                    timeout   = aiohttp.ClientTimeout(total=300, connect=20, sock_read=300)

                    # FIX Bug 1: async with → session pasti ditutup saat selesai, tidak leak
                    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                        pl._http_session = session
                        await run_pipeline(job_id, folder_url)
                    # Setelah context manager keluar, pl._http_session sudah closed
                    # Reset ke None agar tidak ada kode lain yang pakai session stale
                    pl._http_session = None

                new_loop.run_until_complete(_inner())
            finally:
                new_loop.close()

        # FIX Bug 3: Gunakan _pipeline_executor (dedicated) bukan default executor
        # Default executor berbagi dengan Discord heartbeat → bisa starvation
        await loop.run_in_executor(_pipeline_executor, _run_in_thread)

    async def _orchestrate():
        try:
            # FIX: guard kalau on_ready belum sempat init semaphore
            global _global_job_sem
            if _global_job_sem is None:
                _global_job_sem = asyncio.Semaphore(2)

            async with _global_job_sem:
                pipeline_task = asyncio.create_task(_run_pipeline())
                updater_task  = asyncio.create_task(_live_update())
                await pipeline_task

            await asyncio.sleep(2)
            updater_task.cancel()

            try:
                msg   = await interaction.original_response()
                row   = db_get_job(job_id)
                final = dict(row) if row else jobs.get(job_id, {})
                await msg.edit(embed=_make_job_embed(job_id, final))
            except Exception as e:
                print(f"[bot] Gagal update embed final: {e}")

            try:
                await _notify_done(channel, job_id, user.mention)
            except Exception as e:
                print(f"[bot] Gagal kirim notif: {e}")

        except Exception:
            err_tb = traceback.format_exc()
            print(f"[ORCHESTRATE FATAL] job={job_id}:\n{err_tb}")
            try:
                jobs[job_id]["status"] = "failed"
                msg  = await interaction.original_response()
                row  = db_get_job(job_id)
                data = dict(row) if row else jobs.get(job_id, {})
                # Kirim error ke channel agar visible
                short_err = err_tb.strip().split("\n")[-1][:200]
                await channel.send(
                    f"{user.mention} ❌ Job `{job_id}` fatal error:\n```{short_err}```"
                )
                await msg.edit(embed=_make_job_embed(job_id, data))
            except Exception:
                pass
    asyncio.create_task(_orchestrate())


# ── /status ───────────────────────────────────────────────

@tree.command(name="status", description="Cek status job")
@app_commands.describe(job_id="ID job (contoh: a1b2c3d4)")
async def cmd_status(interaction: discord.Interaction, job_id: str):
    job_id = job_id.strip()
    data   = jobs.get(job_id)
    if not data:
        row = db_get_job(job_id)
        if not row:
            await interaction.response.send_message(
                f"❌ Job `{job_id}` tidak ditemukan.", ephemeral=True
            )
            return
        data = dict(row)
    await interaction.response.send_message(embed=_make_job_embed(job_id, data), ephemeral=True)


# ── /cancel ───────────────────────────────────────────────

@tree.command(name="cancel", description="Batalkan job yang sedang berjalan")
@app_commands.describe(job_id="ID job yang ingin dibatalkan")
async def cmd_cancel(interaction: discord.Interaction, job_id: str):
    job_id = job_id.strip()
    data   = jobs.get(job_id)
    status = data.get("status") if data else None
    if not status:
        row    = db_get_job(job_id)
        status = dict(row).get("status") if row else None

    if not status:
        await interaction.response.send_message(
            f"❌ Job `{job_id}` tidak ditemukan.", ephemeral=True
        )
        return

    if status in ("completed", "failed", "cancelled"):
        await interaction.response.send_message(
            f"ℹ️ Job `{job_id}` sudah **{status}**, tidak bisa dibatalkan.",
            ephemeral=True,
        )
        return

    cancelled_jobs.add(job_id)
    await interaction.response.send_message(
        f"🚫 Job `{job_id}` ditandai untuk dibatalkan.",
        ephemeral=True,
    )


# ── Run ───────────────────────────────────────────────────

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)