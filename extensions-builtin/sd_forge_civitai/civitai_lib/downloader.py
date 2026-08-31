"""Background downloader for Civitai models.

Design goals:

* never block the UI - every download runs in its own thread
* never lose work - partial downloads are kept and resumed
* be honest about progress - the UI shows bytes, speed and ETA
* tidy up after itself - metadata (``.json``) and a preview image are written
  next to the model so the file shows up nicely in *Extra Networks*
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid

from civitai_lib import api

ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str, default: str = "model") -> str:
    name = (name or "").strip()
    name = ILLEGAL.sub("_", name)
    name = name.strip(" .")
    return name or default


class DownloadJob:
    def __init__(self, job_id: str, site: str, model_name: str, version_id, file_name: str, target_dir: str, url: str, size: int = 0):
        self.id = job_id
        self.site = site
        self.model_name = model_name
        self.version_id = version_id
        self.file_name = file_name
        self.target_dir = target_dir
        self.url = url
        self.size = size

        self.status = "queued"  # queued | downloading | done | error | cancelled
        self.received = 0
        self.error = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self.path = ""
        self.preview_url = ""
        self.trained_words: list[str] = []
        self.version: dict = {}
        self.model: dict = {}
        # The AutoV2 / SHA256 hashes of the file as published by Civitai. They go
        # into the sidecar so scanning a model folder never has to hash the file
        # again to reach the very same value.
        self.hashes: dict = {}

        self._lock = threading.Lock()
        self._cancel = threading.Event()

    # ---------------------------------------------------------------- progress

    @property
    def speed(self) -> float:
        if not self.started_at:
            return 0.0
        elapsed = (self.finished_at or time.time()) - self.started_at
        return self.received / elapsed if elapsed > 0.05 else 0.0

    @property
    def eta(self) -> float:
        s = self.speed
        if s <= 0 or not self.size:
            return 0.0
        return max(0.0, (self.size - self.received) / s)

    def cancel(self):
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def add(self, n: int):
        with self._lock:
            self.received += n

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "site": self.site,
            "model_name": self.model_name,
            "version_id": self.version_id,
            "file_name": self.file_name,
            "target_dir": self.target_dir,
            "size": self.size,
            "status": self.status,
            "received": self.received,
            "error": self.error,
            "path": self.path,
            "speed": self.speed,
            "eta": self.eta,
        }


class DownloadManager:
    def __init__(self, timeout: float = 30.0):
        self.jobs: dict[str, DownloadJob] = {}
        self.order: list[str] = []
        self.timeout = timeout
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def submit(
        self,
        *,
        site: str,
        api_key: str,
        model: dict,
        version: dict,
        target_dir: str,
        file_name: str | None = None,
        connections: int = 4,
        save_info: bool = True,
        save_preview: bool = True,
        on_finished=None,
    ) -> DownloadJob:
        f = api.primary_file(version) or {}
        name = sanitize_filename(file_name or f.get("name") or f"{model.get('name', 'model')}.safetensors")

        os.makedirs(target_dir, exist_ok=True)

        client = api.CivitaiAPI(site=site, api_key=api_key, timeout=self.timeout)

        # Always build the URL ourselves: the `downloadUrl` shipped in the API response
        # only carries the auth token when the API was called with `?token=`, and we use
        # the `Authorization` header instead.
        url = client.download_url(version.get("id"))

        job = DownloadJob(
            job_id=uuid.uuid4().hex[:12],
            site=site,
            model_name=model.get("name") or "model",
            version_id=version.get("id"),
            file_name=name,
            target_dir=target_dir,
            url=url,
            size=api.file_size_bytes(f),
        )
        job.version = version
        job.model = model
        job.trained_words = api.trained_words(version)
        job.hashes = f.get("hashes") or {}

        images = api.version_images(version, limit=1)
        if images:
            job.preview_url = images[0].get("url", "")

        with self._lock:
            self.jobs[job.id] = job
            self.order.append(job.id)

        thread = threading.Thread(
            target=self._run,
            args=(job, client, connections, save_info, save_preview, on_finished),
            daemon=True,
            name=f"civitai-dl-{job.id}",
        )
        thread.start()
        return job

    def status(self) -> list[dict]:
        with self._lock:
            jobs = [self.jobs[i] for i in self.order if i in self.jobs]
        return [j.as_dict() for j in jobs]

    def cancel(self, job_id: str):
        job = self.jobs.get(job_id)
        if job is not None:
            job.cancel()

    def clear_finished(self):
        with self._lock:
            for job_id in list(self.order):
                job = self.jobs.get(job_id)
                if job is not None and job.status in ("done", "error", "cancelled"):
                    self.order.remove(job_id)
                    self.jobs.pop(job_id, None)

    # ------------------------------------------------------------------ internals

    def _run(self, job: DownloadJob, client: api.CivitaiAPI, connections: int, save_info: bool, save_preview: bool, on_finished):
        job.status = "downloading"
        job.started_at = time.time()

        final_path = os.path.join(job.target_dir, job.file_name)
        part_path = final_path + ".part"

        try:
            total, supports_ranges = self._probe(client, job)

            if total:
                job.size = total

            use_parallel = supports_ranges and total > 4 * 1024 * 1024 and connections > 1

            if use_parallel:
                self._download_parallel(client, job, part_path, total, connections)
            else:
                self._download_single(client, job, part_path, resume=supports_ranges)

            if job.cancelled():
                job.status = "cancelled"
                return

            os.replace(part_path, final_path)
            job.path = final_path

            if save_info:
                self._write_info(final_path, job)
            if save_preview and job.preview_url:
                self._write_preview(client, final_path, job)

            job.status = "done"
            job.finished_at = time.time()

            if on_finished is not None:
                try:
                    on_finished(job)
                except Exception:
                    pass
        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.finished_at = time.time()
        finally:
            if job.status != "done" and os.path.exists(part_path) and job.status == "cancelled":
                try:
                    os.remove(part_path)
                except Exception:
                    pass

    def _probe(self, client: api.CivitaiAPI, job: DownloadJob) -> tuple[int, bool]:
        headers = {}
        if client.api_key:
            headers["Authorization"] = f"Bearer {client.api_key}"

        try:
            r = client.session.head(job.url, headers=headers, timeout=self.timeout, allow_redirects=True)
        except Exception:
            return job.size, False

        if r.status_code >= 400:
            # some CDNs answer HEAD with 403 but serve GET happily
            return job.size, False

        size = 0
        try:
            size = int(r.headers.get("Content-Length", 0) or 0)
        except Exception:
            size = 0

        supports_ranges = r.headers.get("Accept-Ranges", "").lower() == "bytes"
        return size, supports_ranges

    def _open_stream(self, client: api.CivitaiAPI, job: DownloadJob, start: int = 0, end: int | None = None):
        headers = {"User-Agent": "sd-webui-forge-neo/civitai"}
        if client.api_key:
            headers["Authorization"] = f"Bearer {client.api_key}"
        if start or end is not None:
            if end is not None:
                headers["Range"] = f"bytes={start}-{end}"
            else:
                headers["Range"] = f"bytes={start}-"

        r = client.session.get(job.url, headers=headers, stream=True, timeout=(self.timeout, 120))
        if r.status_code >= 400:
            raise api.CivitaiError(f"download failed with {r.status_code} ({job.url.split('?')[0]})")
        return r

    def _download_single(self, client: api.CivitaiAPI, job: DownloadJob, part_path: str, resume: bool):
        start = 0
        if resume and os.path.exists(part_path):
            start = os.path.getsize(part_path)
            # a leftover .part from an interrupted multi-connection run is already
            # pre-allocated to the full size - appending to it would corrupt the file
            if job.size and start >= job.size:
                start = 0
            job.received = start

        r = self._open_stream(client, job, start=start)
        mode = "ab" if start else "wb"

        with open(part_path, mode) as fh:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if job.cancelled():
                    r.close()
                    return
                if not chunk:
                    continue
                fh.write(chunk)
                job.add(len(chunk))

    def _download_parallel(self, client: api.CivitaiAPI, job: DownloadJob, part_path: str, total: int, connections: int):
        if not os.path.exists(part_path) or os.path.getsize(part_path) != total:
            with open(part_path, "wb") as fh:
                try:
                    fh.truncate(total)
                except Exception:
                    pass
        job.received = 0

        chunk = total // connections
        ranges = []
        cursor = 0
        for i in range(connections):
            end = total - 1 if i == connections - 1 else cursor + chunk - 1
            ranges.append((cursor, end))
            cursor = end + 1

        threads = []

        def worker(start, end):
            r = None
            try:
                r = self._open_stream(client, job, start=start, end=end)
                with open(part_path, "r+b") as fh:
                    fh.seek(start)
                    for c in r.iter_content(chunk_size=1024 * 512):
                        if job.cancelled():
                            break
                        if not c:
                            continue
                        fh.write(c)
                        job.add(len(c))
            finally:
                if r is not None:
                    r.close()

        for start, end in ranges:
            if start > end:
                continue
            t = threading.Thread(target=worker, args=(start, end), daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    # ------------------------------------------------------------------ side files

    def _write_info(self, model_path: str, job: DownloadJob):
        base = os.path.splitext(model_path)[0]
        info = {
            "model_name": job.model_name,
            "site": job.site,
            "model_id": (job.model or {}).get("id"),
            "version_id": job.version_id,
            "version_name": (job.version or {}).get("name"),
            "base_model": (job.version or {}).get("baseModel"),
            "type": (job.model or {}).get("type"),
            "creator": ((job.model or {}).get("creator") or {}).get("username"),
            "description": (job.model or {}).get("description") or (job.version or {}).get("description") or "",
            "tags": (job.model or {}).get("tags") or [],
            "trained_words": job.trained_words,
            "download_url": f"https://{job.site}/models/{(job.model or {}).get('id')}?modelVersionId={job.version_id}",
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            # Nested copy for the sidecar readers in the LoRA/embedding scanners:
            # civitai.hashes.AutoV2 is the same value the built-in AutoV2 hash
            # would compute from the file, so it makes scanning instant.
            "civitai": {
                "site": job.site,
                "model_id": (job.model or {}).get("id"),
                "version_id": job.version_id,
                "version_name": (job.version or {}).get("name"),
                "base_model": (job.version or {}).get("baseModel"),
                "hashes": {k: v for k, v in (job.hashes or {}).items() if isinstance(v, str)},
            },
        }

        try:
            with open(base + ".json", "w", encoding="utf8") as fh:
                json.dump(info, fh, indent=4, ensure_ascii=False)
        except Exception:
            pass

        if job.trained_words:
            try:
                with open(base + ".txt", "w", encoding="utf8") as fh:
                    fh.write(", ".join(job.trained_words))
            except Exception:
                pass

    def _write_preview(self, client: api.CivitaiAPI, model_path: str, job: DownloadJob):
        base = os.path.splitext(model_path)[0]
        url = client.thumbnail(job.preview_url, width=512) or job.preview_url

        try:
            content, _ = client.fetch_image(url)
            if not content:
                return
            with open(base + ".png", "wb") as fh:
                fh.write(content)
        except Exception:
            pass
