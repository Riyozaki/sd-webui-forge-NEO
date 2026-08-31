"""Civitai / Civitai Red browser.

Adds a **Civitai** tab that can search, inspect and download models from both

* https://civitai.com (SFW)
* https://civitai.red (age restricted)

Downloads land directly in the matching folder under ``models/`` and get a
``.json`` sidecar plus a preview image, so they immediately show up - with a
thumbnail and their trigger words - in the *Extra Networks* panel.
"""

from __future__ import annotations

import html as html_lib
import os
import sys
import threading
import time

import gradio as gr

from modules import errors, script_callbacks, shared

EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_DIR not in sys.path:
    sys.path.insert(0, EXT_DIR)

from civitai_lib import api  # noqa: E402
from civitai_lib import downloader as dl  # noqa: E402

MANAGER = dl.DownloadManager(timeout=30.0)

FOLDER_MAP = {
    "Checkpoint": "Stable-diffusion",
    "LORA": "Lora",
    "LoCon": "Lora",
    "DoRA": "Lora",
    "TextualInversion": "embeddings",
    "Controlnet": "ControlNet",
    "VAE": "VAE",
    "Upscaler": "ESRGAN",
    "MotionModule": "AnimateDiff",
    "Poses": "Poses",
    "Wildcards": "wildcards",
    "Workflows": "Workflows",
    "Other": "Other",
}


# --------------------------------------------------------------------------- settings helpers


def _opt(name, default=None):
    try:
        return getattr(shared.opts, name, default)
    except Exception:
        return default


def client_for(site: str) -> api.CivitaiAPI:
    site = site or _opt("civitai_default_site", "civitai.com")
    key = _opt("civitai_red_api_key", "") if site == "civitai.red" else _opt("civitai_api_key", "")
    return api.CivitaiAPI(site=site, api_key=key or "", timeout=float(_opt("civitai_timeout", 30) or 30))


def models_root() -> str:
    from modules import paths

    return paths.models_path


def target_folder(model_type: str) -> str:
    folder = FOLDER_MAP.get(model_type, "Other")
    return os.path.join(models_root(), folder)


def image_proxy_url(url: str, site: str, width: int = 320) -> str:
    from urllib.parse import quote

    return f"/civitai/image?site={quote(site)}&w={int(width)}&url={quote(url or '', safe='')}"


def esc(v) -> str:
    return html_lib.escape(str(v if v is not None else ""), quote=True)


def short_number(n) -> str:
    try:
        n = float(n or 0)
    except Exception:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(int(n))


# --------------------------------------------------------------------------- rendering


def render_cards(results: dict, site: str) -> str:
    if not results:
        return '<div class="civitai-empty">Nothing here yet - search for a model to get started.</div>'

    out = ['<div class="civitai-grid">']

    for model_id, model in results.items():
        versions = model.get("modelVersions") or []
        if not versions:
            continue

        first = versions[0]
        images = api.version_images(first, limit=1)
        thumb = api.thumbnail((images[0] if images else {}).get("url", ""), width=320)

        stats = api.stats_of(model)
        nsfw = bool(model.get("nsfw"))
        out.append('<div class="civitai-card" data-civitai-id="%s" onclick="civitaiPickCard(this)">' % esc(model_id))
        out.append('<div class="civitai-thumb">')
        if thumb:
            out.append(f'<img loading="lazy" src="{esc(image_proxy_url(thumb, site))}" alt="">')
        else:
            out.append('<div class="civitai-nothumb">no preview</div>')
        out.append("</div>")
        out.append('<div class="civitai-card-body">')
        out.append(f'<div class="civitai-card-title" title="{esc(model.get("name"))}">{esc(model.get("name"))}</div>')
        out.append('<div class="civitai-badges">')
        out.append(f'<span class="civitai-badge">{esc(model.get("type", "?"))}</span>')
        out.append(f'<span class="civitai-badge civitai-badge-soft">{esc(first.get("baseModel", "?"))}</span>')
        if nsfw:
            out.append('<span class="civitai-badge civitai-badge-nsfw">NSFW</span>')
        if len(versions) > 1:
            out.append(f'<span class="civitai-badge civitai-badge-soft">{len(versions)} versions</span>')
        out.append("</div>")
        out.append(
            '<div class="civitai-stats"><span title="rating">&#9733; %s</span><span title="downloads">&#8681; %s</span></div>'
            % (esc(round(float(stats["rating"] or 0), 1)), esc(short_number(stats["downloads"])))
        )
        out.append("</div>")  # body
        out.append("</div>")  # card

    out.append("</div>")
    return "".join(out)


def render_details(model: dict | None, version: dict | None, site: str) -> str:
    if not model:
        return '<div class="civitai-empty">Select a card to see its details.</div>'

    stats = api.stats_of(model)
    creator = (model.get("creator") or {}).get("username", "unknown")
    tags = [t for t in (model.get("tags") or []) if isinstance(t, str)][:24]

    parts = ['<div class="civitai-details">']
    parts.append(f'<div class="civitai-details-title">{esc(model.get("name"))}</div>')
    parts.append(
        f'<div class="civitai-details-sub">by <b>{esc(creator)}</b> &middot; '
        f'{esc(model.get("type", "?"))} &middot; &#9733; {esc(round(float(stats["rating"] or 0), 2))} '
        f'({esc(short_number(stats["rating_count"]))} ratings) &middot; '
        f'&#8681; {esc(short_number(stats["downloads"]))}</div>'
    )
    parts.append(
        f'<div class="civitai-details-links"><a href="https://{esc(site)}/models/{esc(model.get("id"))}" target="_blank" rel="noreferrer">Open on {esc(site)}</a></div>'
    )

    if tags:
        parts.append('<div class="civitai-badges">')
        for t in tags:
            parts.append(f'<span class="civitai-badge civitai-badge-soft">{esc(t)}</span>')
        parts.append("</div>")

    description = (model.get("description") or "").strip()
    if description:
        keep = description[:1400]
        if len(description) > 1400:
            keep += " ..."
        parts.append(f'<div class="civitai-description">{esc(keep)}</div>')

    if version is not None:
        f = api.primary_file(version) or {}
        words = api.trained_words(version)
        parts.append('<div class="civitai-version">')
        parts.append(
            f'<div><b>{esc(version.get("name") or "unnamed")}</b> &middot; {esc(version.get("baseModel", "?"))} '
            f'&middot; {esc(api.human_size(api.file_size_bytes(f)))} &middot; {esc(f.get("format") or f.get("name") or "?")}</div>'
        )
        if words:
            parts.append('<div class="civitai-badges">')
            for w in words[:20]:
                parts.append(f'<span class="civitai-badge civitai-badge-word">{esc(w)}</span>')
            parts.append("</div>")
            parts.append(
                '<div class="civitai-hint"><a onclick="civitaiCopyWords(this)" data-words="%s" href="#">copy trigger words to the positive prompt</a></div>'
                % esc(", ".join(words))
            )
        published = version.get("publishedAt") or version.get("createdAt") or ""
        if published:
            parts.append(f'<div class="civitai-hint">published {esc(str(published)[:10])}</div>')
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def render_downloads() -> str:
    jobs = MANAGER.status()

    if not jobs:
        return '<div class="civitai-empty">No downloads yet.</div>'

    out = ['<div class="civitai-downloads">']
    for job in reversed(jobs[-40:]):
        percent = 0.0
        if job["size"]:
            percent = min(1.0, job["received"] / job["size"])
        elif job["status"] == "done":
            percent = 1.0

        status = job["status"]
        out.append('<div class="civitai-download">')
        out.append('<div class="civitai-download-head">')
        out.append(f'<span class="civitai-download-name">{esc(job["file_name"])}</span>')
        out.append(f'<span class="civitai-download-status civitai-status-{esc(status)}">{esc(status)}</span>')
        out.append("</div>")
        out.append(f'<div class="civitai-bar"><div class="civitai-bar-fill" style="width:{percent * 100:.1f}%"></div></div>')
        out.append('<div class="civitai-download-meta">')
        out.append(
            f'<span>{esc(api.human_size(job["received"]))} / {esc(api.human_size(job["size"]))}</span>'
        )
        if status == "downloading":
            out.append(
                f'<span>{esc(api.human_size(job["speed"]))}/s &middot; ETA {esc(int(job["eta"]))}s</span>'
            )
        if job["error"]:
            out.append(f'<span class="civitai-error">{esc(job["error"])}</span>')
        if job["path"]:
            out.append(f'<span class="civitai-path">{esc(job["path"])}</span>')
        out.append("</div>")
        out.append("</div>")

    out.append("</div>")
    return "".join(out)


# --------------------------------------------------------------------------- update check

# Comparing every installed network with the site takes two requests each, so
# the check runs in a thread and the tab polls for its state.
UPDATE_STATE: dict = {
    "running": False,
    "checked": 0,
    "total": 0,
    "skipped": 0,
    "items": {},
    "error": "",
    "finished_at": 0.0,
}

UPDATE_HELP = (
    '<div class="civitai-hint">Compares the hash of every installed LoRA with '
    "Civitai. Nothing is downloaded and no file is hashed again &mdash; networks "
    "Forge has not hashed yet are skipped, so open the <i>Extra Networks</i> tab "
    "once if they are missing.</div>"
)


def lora_networks() -> tuple[list[dict], int]:
    """Installed networks, split into 'has a hash' and 'has not', without hashing.

    The hash Civitai indexes is the same AutoV2 value Forge already stores for
    every network, so an update check never touches the disk.
    """
    import sys

    module = sys.modules.get("networks")
    if module is None:
        try:
            from modules import paths_internal

            scripts = os.path.join(paths_internal.extensions_builtin_dir, "sd_forge_lora", "scripts")
            if os.path.isdir(scripts) and scripts not in sys.path:
                sys.path.insert(0, scripts)
            import networks as module  # noqa: F811
        except Exception:
            return [], 0

    available = getattr(module, "available_networks", None) or {}

    known: list[dict] = []
    total = 0
    for name, net in available.items():
        total += 1
        file_hash = str(getattr(net, "hash", "") or "").strip()
        if len(file_hash) != 64:
            continue
        known.append(
            {
                "name": str(getattr(net, "alias", "") or name),
                "file": str(getattr(net, "filename", "") or ""),
                "hash": file_hash,
            }
        )

    known.sort(key=lambda entry: entry["name"].lower())
    return known, total


def render_updates_status() -> str:
    state = UPDATE_STATE

    if state["running"]:
        text = f'Checking {state["checked"]} of {state["total"]} installed networks&hellip;'
        return f'<div class="civitai-progress" data-running="1">{text}</div>'

    if state["error"]:
        return f'<div class="civitai-error-box" data-running="0">{esc(state["error"])}</div>'

    if not state["finished_at"]:
        return '<div class="civitai-empty" data-running="0">Not checked yet.</div>'

    updates = len(state["items"])
    when = time.strftime("%H:%M:%S", time.localtime(state["finished_at"]))
    parts = [f'{state["total"]} networks checked at {esc(when)}']
    parts.append(f'<b class="civitai-updates-found">{updates} update(s)</b>' if updates else "everything is up to date")
    if state["skipped"]:
        parts.append(f'{state["skipped"]} without a hash yet')
    return f'<div class="civitai-progress" data-running="0">{" &middot; ".join(parts)}</div>'


def render_updates_list(site: str = "civitai.com") -> str:
    state = UPDATE_STATE

    if not state["items"]:
        if state["running"]:
            return '<div class="civitai-empty">Looking&hellip;</div>'
        if state["finished_at"]:
            return '<div class="civitai-empty">Every network is on its newest version.</div>'
        return '<div class="civitai-empty">Press <b>Check for updates</b> to compare your LoRAs with Civitai.</div>'

    out = ['<div class="civitai-updates">']
    for file_hash, entry in sorted(state["items"].items(), key=lambda kv: str((kv[1] or {}).get("name", "")).lower()):
        installed = entry.get("installed") or {}
        latest = entry.get("latest") or {}
        newer = entry.get("newer") or []

        model_id = entry.get("model_id")
        link = f"https://{site}/models/{model_id}?modelVersionId={latest.get('id')}" if model_id else ""

        out.append('<div class="civitai-update">')
        out.append('<div class="civitai-update-head">')
        out.append(f'<span class="civitai-update-name">{esc(entry.get("name") or "?")}</span>')
        current_name = installed.get("name") or "?"
        new_name = latest.get("name") or "?"
        out.append(
            '<span class="civitai-update-versions">'
            f"{esc(current_name)} &rarr; <b>{esc(new_name)}</b>"
            + (f' <span class="civitai-update-count">+{len(newer) - 1} more version(s)</span>' if len(newer) > 1 else "")
            + "</span>"
        )
        out.append("</div>")
        out.append('<div class="civitai-update-meta">')
        out.append(f"<span>{esc(str(latest.get('baseModel') or ''))}</span>")
        published = str(latest.get("publishedAt") or latest.get("createdAt") or "")[:10]
        if published:
            out.append(f"<span>published {esc(published)}</span>")
        if entry.get("file"):
            out.append(f'<span class="civitai-path">{esc(os.path.basename(entry["file"]))}</span>')
        if link:
            out.append(f'<a href="{esc(link)}" target="_blank" rel="noreferrer">open on {esc(site)}</a>')
        out.append("</div>")
        out.append(
            f'<button class="civitai-update-btn" data-hash="{esc(file_hash)}" '
            'onclick="civitaiUpdate(this)">Download the new version</button>'
        )
        out.append("</div>")

    out.append("</div>")
    return "".join(out)


def _run_update_check(site: str, entries: list[dict]):
    client = client_for(site)
    last = 0.0
    failures = 0

    for entry in entries:
        if not UPDATE_STATE["running"]:
            break
        try:
            last = api.rate_limit_sleep(last, 0.25)
            info = client.update_info(entry["hash"])
            failures = 0
        except Exception as e:
            failures += 1
            UPDATE_STATE["error"] = str(e)
            if failures >= 3:
                break
            continue

        UPDATE_STATE["checked"] += 1
        if info and info.get("newer"):
            UPDATE_STATE["items"][entry["hash"]] = dict(info, file=entry["file"], name=entry["name"])

    UPDATE_STATE["running"] = False
    UPDATE_STATE["finished_at"] = time.time()


def do_check_updates(site: str):
    if UPDATE_STATE["running"]:
        return render_updates_status(), render_updates_list(site)

    entries, total = lora_networks()
    UPDATE_STATE.update(
        {
            "running": False,
            "checked": 0,
            "total": total,
            "skipped": max(0, total - len(entries)),
            "items": {},
            "error": "",
            "finished_at": 0.0,
        }
    )

    if not entries:
        UPDATE_STATE["error"] = (
            "None of the installed networks has a hash yet. Open the Extra Networks "
            "tab once so Forge can compute them, then check again."
        )
        return render_updates_status(), render_updates_list(site)

    UPDATE_STATE["running"] = True
    threading.Thread(target=_run_update_check, args=(site, entries), daemon=True, name="civitai-update-check").start()
    return render_updates_status(), render_updates_list(site)


def do_stop_update_check():
    UPDATE_STATE["running"] = False
    return render_updates_status()


def poll_updates(site: str):
    return render_updates_status(), render_updates_list(site)


def apply_update(pick: str, site: str):
    entry = UPDATE_STATE["items"].get(str(pick or "").strip())
    if entry is None:
        return '<div class="civitai-error-box" data-running="0">That update is gone - check again.</div>', render_downloads()

    latest = entry.get("latest") or {}
    model = {
        "id": entry.get("model_id"),
        "name": entry.get("model_name") or entry.get("name"),
        "type": entry.get("model_type") or "LORA",
    }

    folder = os.path.dirname(entry.get("file") or "") or target_folder(model["type"])

    try:
        client = client_for(site)
        MANAGER.submit(
            site=site,
            api_key=client.api_key,
            model=model,
            version=latest,
            target_dir=folder,
            connections=int(_opt("civitai_download_threads", 4) or 4),
            save_info=bool(_opt("civitai_save_info", True)),
            save_preview=bool(_opt("civitai_save_preview", True)),
            on_finished=lambda j: refresh_model_lists(model["type"]),
        )
    except Exception as e:
        errors.display(e, "civitai update download")
        return f'<div class="civitai-error-box" data-running="0">{esc(e)}</div>', render_downloads()

    message = (
        '<div class="civitai-progress" data-running="0">Downloading '
        f'{esc(latest.get("name") or "the new version")} of {esc(model["name"] or "?")} '
        "&mdash; see <b>Downloads</b> below.</div>"
    )
    return message, render_downloads()


# --------------------------------------------------------------------------- actions


def do_search(site, query, model_type, base_model, sort, period, nsfw, limit, results):
    try:
        client = client_for(site)
        data = client.search(
            query=query or "",
            types=model_type,
            base_models=base_model,
            sort=sort,
            period=period,
            nsfw=bool(nsfw) or site == "civitai.red",
            limit=int(limit or 24),
        )
    except Exception as e:
        errors.display(e, "civitai search")
        return f'<div class="civitai-error-box">{esc(e)}</div>', results, "search failed"

    items = [i for i in (data.get("items") or []) if isinstance(i, dict) and i.get("modelVersions")]
    new_results = {str(i.get("id")): i for i in items}

    meta = data.get("metadata") or {}
    total = meta.get("totalItems")
    info = f"{len(items)} of {short_number(total)}" if total else f"{len(items)} results"

    return render_cards(new_results, site), new_results, info


def do_lookup(site, text, results):
    text = (text or "").strip()
    if not text:
        return "Paste a Civitai URL or a model hash first.", results

    try:
        client = client_for(site)

        model_id, version_id = api.parse_civitai_url(text)

        if model_id is None and version_id is None and len(text) in (10, 64):
            # looks like an AutoV2 (10) or SHA256 (64) hash
            version = client.version_by_hash(text)
            if version is None:
                return "Civitai does not know this hash.", results
            model_id = (version.get("model") or {}).get("id", version.get("modelId"))
            version_id = version.get("id")

        if model_id is None and version_id is not None:
            version = client.get_model_version(version_id)
            model_id = (version.get("model") or {}).get("id", version.get("modelId"))

        if model_id is None:
            return "Could not understand that URL/hash.", results

        model = client.get_model(model_id)
    except Exception as e:
        errors.display(e, "civitai lookup")
        return f'<div class="civitai-error-box">{esc(e)}</div>', results

    new_results = {str(model.get("id")): model}
    return "found: " + esc(model.get("name")), new_results


def on_select(model_id, site, results):
    model = (results or {}).get(str(model_id))
    if not model:
        return render_details(None, None, site), gr.update(), gr.update()

    versions = [v for v in (model.get("modelVersions") or []) if isinstance(v, dict)]
    choices = [(api.version_label(v), str(v.get("id"))) for v in versions]

    return (
        render_details(model, versions[0] if versions else None, site),
        gr.update(choices=choices, value=choices[0][1] if choices else None),
        {str(v.get("id")): v for v in versions},
    )


def on_version_change(version_id, model_id, site, results, versions):
    model = (results or {}).get(str(model_id))
    version = (versions or {}).get(str(version_id))
    return render_details(model, version, site)


def refresh_model_lists(model_type: str):
    try:
        if model_type == "Checkpoint":
            from modules import sd_models

            sd_models.list_models()
        elif model_type in ("LORA", "LoCon", "DoRA"):
            import lora

            lora.list_available_loras()
    except Exception as e:
        errors.display(e, "civitai: refreshing model lists")


def on_download(model_id, version_id, site, results, versions):
    model = (results or {}).get(str(model_id))
    version = (versions or {}).get(str(version_id))

    if not model or not version:
        yield "Select a model and a version first.", render_downloads()
        return

    client = client_for(site)
    model_type = model.get("type") or "Other"
    folder = target_folder(model_type)

    try:
        job = MANAGER.submit(
            site=site,
            api_key=client.api_key,
            model=model,
            version=version,
            target_dir=folder,
            connections=int(_opt("civitai_download_threads", 4) or 4),
            save_info=bool(_opt("civitai_save_info", True)),
            save_preview=bool(_opt("civitai_save_preview", True)),
            on_finished=lambda j: refresh_model_lists(model_type),
        )
    except Exception as e:
        errors.display(e, "civitai download")
        yield f'<div class="civitai-error-box">{esc(e)}</div>', render_downloads()
        return

    deadline = time.time() + 60 * 60 * 6

    while True:
        d = job.as_dict()
        if d["size"]:
            ratio = min(1.0, d["received"] / d["size"])
        else:
            ratio = 0.0

        if d["status"] == "downloading":
            text = (
                f'<div class="civitai-progress">Downloading <b>{esc(d["file_name"])}</b><br>'
                f'{esc(api.human_size(d["received"]))} / {esc(api.human_size(d["size"]))} '
                f'({ratio * 100:.1f}%) &middot; {esc(api.human_size(d["speed"]))}/s'
                + (f' &middot; ETA {int(d["eta"])}s' if d["eta"] else "")
                + "</div>"
            )
        else:
            text = f'<div class="civitai-progress">{esc(d["file_name"])}: <b>{esc(d["status"])}</b> {esc(d["error"])}</div>'

        yield text, render_downloads()

        if d["status"] in ("done", "error", "cancelled"):
            break
        if time.time() > deadline:
            break

        time.sleep(0.4)


# --------------------------------------------------------------------------- image proxy


def civitai_image(site: str = "civitai.com", url: str = "", w: int = 320):
    from fastapi.responses import Response

    if not url:
        return Response(status_code=404)

    cache_root = _image_cache_dir()
    key = api_hash(site, url, w)
    path = os.path.join(cache_root, key + ".img")

    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                return Response(content=fh.read(), media_type="image/jpeg")
        except Exception:
            pass

    try:
        client = client_for(site)
        target = client.thumbnail(url, width=int(w or 320)) or url
        content, content_type = client.fetch_image(target)
    except Exception:
        return Response(status_code=404)

    try:
        with open(path, "wb") as fh:
            fh.write(content)
    except Exception:
        pass

    return Response(content=content, media_type=content_type or "image/jpeg")


def api_hash(site: str, url: str, w: int) -> str:
    import hashlib

    return hashlib.sha1(f"{site}|{w}|{url}".encode("utf8")).hexdigest()[:32]


def _image_cache_dir() -> str:
    from modules import paths

    d = os.path.join(paths.data_path, "cache", "civitai-images")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


# --------------------------------------------------------------------------- UI


def on_ui_tabs():
    default_site = _opt("civitai_default_site", "civitai.com")

    with gr.Blocks(analytics_enabled=False) as civitai_interface:
        gr.HTML(
            '<div class="civitai-header">Browse and download models from '
            '<a href="https://civitai.com" target="_blank" rel="noreferrer">Civitai</a> and '
            '<a href="https://civitai.red" target="_blank" rel="noreferrer">Civitai Red</a>. '
            "API keys are configured in <b>Settings &rarr; Neo Optimizations &rarr; Civitai</b>.</div>"
        )

        results_state = gr.State({})
        versions_state = gr.State({})

        with gr.Row():
            site = gr.Radio(label="Site", choices=list(api.SITES), value=default_site, elem_id="civitai_site")
            query = gr.Textbox(label="Search", placeholder="model name, tag, creator...", scale=3, elem_id="civitai_query")
            search_btn = gr.Button("Search", variant="primary", elem_id="civitai_search")
            lookup_btn = gr.Button("Open URL / hash", elem_id="civitai_lookup")

        with gr.Row():
            model_type = gr.Dropdown(label="Type", choices=api.MODEL_TYPES, value="Checkpoint", elem_id="civitai_type")
            base_model = gr.Dropdown(label="Base model", choices=api.BASE_MODELS, value="Any", elem_id="civitai_base")
            sort = gr.Dropdown(label="Sort", choices=api.SORT_ORDERS, value="Most Downloaded", elem_id="civitai_sort")
            period = gr.Dropdown(label="Period", choices=api.PERIODS, value="AllTime", elem_id="civitai_period")
            nsfw = gr.Checkbox(label="NSFW", value=bool(_opt("civitai_allow_nsfw", False)), elem_id="civitai_nsfw")
            limit = gr.Slider(label="Results", minimum=8, maximum=100, step=4, value=24, elem_id="civitai_limit")

        status = gr.HTML('<div class="civitai-empty">Ready.</div>', elem_id="civitai_status")

        selected_model = gr.Textbox(visible=False, elem_id="civitai_selected_model")
        # A hidden button is the most reliable way to wake Python up from JS: the
        # click submits the current (just updated) value of the textbox above.
        select_btn = gr.Button(visible=False, elem_id="civitai_select_btn")

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                cards = gr.HTML('<div class="civitai-empty">Nothing here yet - search for a model to get started.</div>', elem_id="civitai_cards")

            with gr.Column(scale=2):
                details = gr.HTML('<div class="civitai-empty">Select a card to see its details.</div>', elem_id="civitai_details")
                version_dd = gr.Dropdown(label="Version", choices=[], value=None, elem_id="civitai_version")
                with gr.Row():
                    download_btn = gr.Button("Download", variant="primary", elem_id="civitai_download")
                progress_html = gr.HTML("", elem_id="civitai_progress")

        with gr.Accordion("Downloads", open=True, elem_id="civitai_downloads_accordion"):
            downloads = gr.HTML(render_downloads(), elem_id="civitai_downloads")
            with gr.Row():
                refresh_dl = gr.Button("Refresh", elem_id="civitai_refresh_dl")
                clear_dl = gr.Button("Clear finished", elem_id="civitai_clear_dl")

        with gr.Accordion("Updates for installed LoRAs", open=False, elem_id="civitai_updates_accordion"):
            gr.HTML(UPDATE_HELP)
            updates_status = gr.HTML(render_updates_status(), elem_id="civitai_updates_status")
            updates_list = gr.HTML(render_updates_list(default_site), elem_id="civitai_updates_list")
            update_pick = gr.Textbox(visible=False, elem_id="civitai_update_pick")
            update_pick_btn = gr.Button(visible=False, elem_id="civitai_update_pick_btn")
            updates_poll = gr.Button(visible=False, elem_id="civitai_updates_poll")
            with gr.Row():
                check_updates_btn = gr.Button("Check for updates", variant="primary", elem_id="civitai_update_check")
                stop_updates_btn = gr.Button("Stop", elem_id="civitai_update_stop")

        search_btn.click(
            fn=do_search,
            inputs=[site, query, model_type, base_model, sort, period, nsfw, limit, results_state],
            outputs=[cards, results_state, status],
        )
        query.submit(
            fn=do_search,
            inputs=[site, query, model_type, base_model, sort, period, nsfw, limit, results_state],
            outputs=[cards, results_state, status],
        )
        lookup_btn.click(fn=do_lookup, inputs=[site, query, results_state], outputs=[status, results_state]).then(
            fn=lambda r, s: render_cards(r or {}, s), inputs=[results_state, site], outputs=[cards]
        )

        select_btn.click(
            fn=on_select,
            inputs=[selected_model, site, results_state],
            outputs=[details, version_dd, versions_state],
        )
        version_dd.change(
            fn=on_version_change,
            inputs=[version_dd, selected_model, site, results_state, versions_state],
            outputs=[details],
        )

        download_btn.click(
            fn=on_download,
            inputs=[selected_model, version_dd, site, results_state, versions_state],
            outputs=[progress_html, downloads],
        )

        refresh_dl.click(fn=render_downloads, inputs=[], outputs=[downloads])
        clear_dl.click(fn=lambda: (MANAGER.clear_finished(), render_downloads())[1], inputs=[], outputs=[downloads])

        check_updates_btn.click(fn=do_check_updates, inputs=[site], outputs=[updates_status, updates_list])
        stop_updates_btn.click(fn=do_stop_update_check, inputs=[], outputs=[updates_status])
        updates_poll.click(fn=poll_updates, inputs=[site], outputs=[updates_status, updates_list])
        update_pick_btn.click(fn=apply_update, inputs=[update_pick, site], outputs=[updates_status, downloads])

    return [(civitai_interface, "Civitai", "civitai")]


def on_app_started(demo, app):
    try:
        app.add_api_route("/civitai/image", civitai_image, methods=["GET"])
    except Exception as e:
        errors.display(e, "civitai: registering /civitai/image")


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(on_app_started)
