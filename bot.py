# ============================================================
# bot.py — Discord Bot untuk auto_celan
#
# Cara run:
#   python bot.py
#
# Commands:
#   /clean   <folder_url>  — mulai proses folder Drive
#   /status  <job_id>      — cek status job
#   /cancel  <job_id>      — batalkan job yang sedang jalan
#   /help                  — daftar command
# ============================================================

import asyncio
import os
import sys
import socket
import uuid

import discord
from discord import app_commands
import aiohttp
from google.cloud import vision
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────

DISCORD_TOKEN       = os.getenv("DISCORD_BOT_TOKEN")        # token bot BARU
ERROR_CHANNEL_ID    = int(os.getenv("ERROR_CHANNEL_ID", "0"))

if not DISCORD_TOKEN:
    print("❌ DISCORD_BOT_TOKEN tidak ditemukan di .env")
    sys.exit(1)

# ── Import core modules ───────────────────────────────────

from core.config     import GOOGLE_API_KEY
from core.database   import (
    init_db, db_create_job, db_get_job,
    db_register_user, db_update_last_job,
)
from core.drive      import extract_folder_id, filter_and_sort_files, _get_drive
import core.runpod_client as runpod_module
import core.pipeline      as pipeline_module
from core.pipeline   import jobs, cancelled_jobs, run_pipeline, warmup, deletion_loop


# ── Helpers ───────────────────────────────────────────────

def _make_job_embed(job_id: str, data: dict, color=None) -> discord.Embed:
    """Buat Discord Embed dari data job."""
    status = data.get("status", "?")

    color_map = {
        "queued":    discord.Color.yellow(),
        "running":   discord.Color.blue(),
        "completed": discord.Color.green(),
        "failed":    discord.Color.red(),
        "cancelled": discord.Color.grayed(),
    }
    embed_color = color or color_map.get(status, discord.Color.blurple())

    status_icon = {
        "queued":    "⏳",
        "running":   "⚙️",
        "completed": "✅",
        "failed":    "❌",
        "cancelled": "🚫",
    }.get(status, "❓")

    embed = discord.Embed(
        title=f"{status_icon} Job `{job_id}` — {status.upper()}",
        color=embed_color,
    )

    total     = data.get("total_files")    or 0
    completed = data.get("completed_files") or 0
    failed    = data.get("failed_files")
    eta       = data.get("eta")
    progress  = data.get("progress", "")
    result    = data.get("result_folder")
    output_id = data.get("output_folder_id")

    if total > 0:
        pct  = int(completed / total * 100)
        bar  = "█" * (pct // 10) + "░" * (10 - pct // 10)
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

    if status == "completed" and result:
        folder_url = (
            f"https://drive.google.com/drive/folders/{output_id}"
            if output_id else "—"
        )
        embed.add_field(name="📁 Output Drive", value=folder_url, inline=False)

    if failed and failed != "[]":
        import json
        failed_list = json.loads(failed) if isinstance(failed, str) else failed
        if failed_list:
            preview = ", ".join(failed_list[:5])
            if len(failed_list) > 5:
                preview += f" (+{len(failed_list)-5} lagi)"
            embed.add_field(name="❌ Gagal", value=preview, inline=False)

    return embed


async def _notify_done(channel: discord.TextChannel, job_id: str, mention: str):
    """Kirim notifikasi selesai ke channel dengan mention user."""
    row = db_get_job(job_id)
    if not row:
        return

    data   = dict(row)
    status = data.get("status", "?")
    embed  = _make_job_embed(job_id, data)

    icon = "✅" if status == "completed" else "❌"
    await channel.send(
        content=f"{mention} {icon} Job `{job_id}` selesai dengan status **{status}**!",
        embed=embed,
    )


# ── Bot Setup ─────────────────────────────────────────────

intents         = discord.Intents.default()
bot             = discord.Client(intents=intents)
tree            = app_commands.CommandTree(bot)


@bot.event
async def on_ready():
    # Sync slash commands ke Discord
    await tree.sync()
    print(f"✅ Bot online sebagai {bot.user} (ID: {bot.user.id})")
    print(f"   Slash commands sudah di-sync.")

    # Init DB & pipeline state
    init_db()

    runpod_module.runpod_sem     = asyncio.Semaphore(8)
    pipeline_module._warmup_lock = asyncio.Lock()
    pipeline_module.pipeline_sem = asyncio.Semaphore(10)
    pipeline_module.vision_sem   = asyncio.Semaphore(5)

    timeout   = aiohttp.ClientTimeout(total=300, connect=20, sock_connect=20, sock_read=300)
    connector = aiohttp.TCPConnector(limit=30, ttl_dns_cache=300, family=socket.AF_INET)
    pipeline_module._http_session = aiohttp.ClientSession(
        timeout=timeout, connector=connector
    )
    pipeline_module.vision_client = vision.ImageAnnotatorClient(
        client_options={"api_key": GOOGLE_API_KEY}
    )

    # Background tasks
    asyncio.create_task(deletion_loop())
    asyncio.create_task(warmup())


@bot.event
async def on_close():
    if pipeline_module._http_session:
        await pipeline_module._http_session.close()


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
    embed.add_field(
        name="`/status <job_id>`",
        value="Cek status job yang sedang atau sudah selesai.",
        inline=False,
    )
    embed.add_field(
        name="`/cancel <job_id>`",
        value="Batalkan job yang sedang berjalan.",
        inline=False,
    )
    embed.set_footer(text="Bot akan mention kamu otomatis saat job selesai.")
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ── /clean ────────────────────────────────────────────────

@tree.command(
    name="clean",
    description="Proses folder Google Drive — hapus teks manga",
)
@app_commands.describe(folder_url="Link Google Drive folder berisi gambar manga")
async def cmd_clean(interaction: discord.Interaction, folder_url: str):
    await interaction.response.defer(thinking=True)

    user       = interaction.user
    channel    = interaction.channel
    discord_id = str(user.id)
    username   = str(user)

    # Daftarkan user kalau belum ada
    db_register_user(discord_id, username)
    db_update_last_job(discord_id)

    # Validasi folder ID
    folder_id = extract_folder_id(folder_url)
    if not folder_id or len(folder_id) < 10:
        await interaction.followup.send(
            "❌ Link Google Drive tidak valid. Pastikan formatnya:\n"
            "`https://drive.google.com/drive/folders/FOLDER_ID`",
            ephemeral=True,
        )
        return

    # Cek folder bisa diakses
    try:
        def _check():
            return _get_drive().files().get(
                fileId=folder_id, fields="id,name"
            ).execute()
        meta = await asyncio.to_thread(_check)
    except Exception as e:
        await interaction.followup.send(
            f"❌ Folder tidak bisa diakses: `{e}`\n"
            "Pastikan folder sudah di-share ke akun Drive bot.",
            ephemeral=True,
        )
        return

    # Buat job
    job_id = str(uuid.uuid4())[:8]
    db_create_job(job_id, discord_id, folder_url, 0)
    jobs[job_id] = {
        "status":          "queued",
        "progress":        "Starting...",
        "result_folder":   None,
        "output_folder_id": None,
        "queue_position":  0,
        "total_files":     0,
        "completed_files": 0,
        "failed_files":    [],
        "eta":             "...",
        "log":             [],
    }

    # Kirim pesan awal
    embed = discord.Embed(
        title=f"⏳ Job `{job_id}` dimulai",
        description=f"Folder: **{meta.get('name', folder_id)}**",
        color=discord.Color.yellow(),
    )
    embed.add_field(name="Status", value="Sedang antri / mulai...", inline=False)
    embed.set_footer(text="Kamu akan di-mention saat selesai.")
    await interaction.followup.send(embed=embed)

    # Jalankan pipeline di background, notify saat selesai
    async def _run_and_notify():
        # Update embed tiap 5 detik selama proses berjalan
        async def _live_update():
            first_msg = await interaction.original_response()
            while True:
                await asyncio.sleep(5)
                fresh = jobs.get(job_id) or {}
                status = fresh.get("status", "?")
                try:
                    await first_msg.edit(embed=_make_job_embed(job_id, fresh))
                except Exception:
                    pass
                if status in ("completed", "failed", "cancelled"):
                    break

        updater = asyncio.create_task(_live_update())
        # FIX: jalankan pipeline di executor agar tidak blokir event loop Discord
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: asyncio.run(run_pipeline(job_id, folder_url)))
        updater.cancel()

        try:
            await _notify_done(channel, job_id, user.mention)
        except Exception as e:
            print(f"[bot] Gagal kirim notif selesai: {e}")

    asyncio.create_task(_run_and_notify())


# ── /status ───────────────────────────────────────────────

@tree.command(name="status", description="Cek status job")
@app_commands.describe(job_id="ID job yang ingin dicek (contoh: a1b2c3d4)")
async def cmd_status(interaction: discord.Interaction, job_id: str):
    job_id = job_id.strip()

    # Cek memori dulu, fallback ke DB
    data = jobs.get(job_id)
    if not data:
        row = db_get_job(job_id)
        if not row:
            await interaction.response.send_message(
                f"❌ Job `{job_id}` tidak ditemukan.", ephemeral=True
            )
            return
        data = dict(row)

    embed = _make_job_embed(job_id, data)
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ── /cancel ───────────────────────────────────────────────

@tree.command(name="cancel", description="Batalkan job yang sedang berjalan")
@app_commands.describe(job_id="ID job yang ingin dibatalkan")
async def cmd_cancel(interaction: discord.Interaction, job_id: str):
    job_id = job_id.strip()

    data = jobs.get(job_id)
    if not data:
        row = db_get_job(job_id)
        if not row:
            await interaction.response.send_message(
                f"❌ Job `{job_id}` tidak ditemukan.", ephemeral=True
            )
            return
        status = dict(row).get("status")
    else:
        status = data.get("status")

    if status in ("completed", "failed", "cancelled"):
        await interaction.response.send_message(
            f"ℹ️ Job `{job_id}` sudah dalam status **{status}**, tidak bisa dibatalkan.",
            ephemeral=True,
        )
        return

    cancelled_jobs.add(job_id)
    await interaction.response.send_message(
        f"🚫 Job `{job_id}` ditandai untuk dibatalkan. "
        "Akan berhenti setelah file yang sedang diproses selesai.",
        ephemeral=True,
    )


# ── Run ───────────────────────────────────────────────────

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
