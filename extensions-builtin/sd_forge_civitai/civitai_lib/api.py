"""Thin, defensive wrapper around the Civitai REST API (v1).

The Civitai API is not perfectly stable and rate limits aggressively, so every
call is retried with backoff and every field is read with ``.get()``: a missing
key must never break the UI.
"""

from __future__ import annotations

import threading
import time
from urllib.parse import quote, urlencode

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 < 2 keeps Retry in `urllib3.util.retry`
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

SITES = ("civitai.com", "civitai.red")

BASE_MODELS = [
    "Any",
    "SD 1.4",
    "SD 1.5",
    "SD 2.0",
    "SD 2.1",
    "SD 3",
    "SD 3.5",
    "SDXL 0.9",
    "SDXL 1.0",
    "Pony",
    "Flux.1 S",
    "Flux.1 D",
    "Illustrious",
    "Other",
]

MODEL_TYPES = [
    "Any",
    "Checkpoint",
    "LORA",
    "LoCon",
    "DoRA",
    "TextualInversion",
    "Controlnet",
    "VAE",
    "Upscaler",
    "MotionModule",
    "Poses",
    "Wildcards",
    "Workflows",
    "Other",
]

SORT_ORDERS = [
    "Most Downloaded",
    "Highest Rated",
    "Most Liked",
    "Most Discussed",
    "Most Collected",
    "Newest",
]

PERIODS = ["AllTime", "Year", "Month", "Week", "Day"]


class CivitaiError(Exception):
    """Anything that went wrong while talking to Civitai."""


def thumbnail(url: str, width: int = 320) -> str | None:
    """Ask Civitai's image CDN for a smaller version of a preview image.

    ``image.civitai.com`` URLs embed the requested width (``/width=450/``), so we
    can simply rewrite it instead of downloading the full sized image.
    """
    if not url:
        return None

    out = url
    if "/width=" in out:
        head, _, tail = out.partition("/width=")
        _, _, rest = tail.partition("/")
        out = f"{head}/width={width}/{rest}"
    return out


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


class CivitaiAPI:
    def __init__(self, site: str = "civitai.com", api_key: str = "", timeout: float = 30.0):
        self.site = site if site in SITES else "civitai.com"
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self._sessions = threading.local()
        self._cache_lock = threading.Lock()
        self._cache: dict = {}

    # ------------------------------------------------------------------ plumbing

    @property
    def host(self) -> str:
        return f"https://{self.site}"

    @property
    def base(self) -> str:
        return f"{self.host}/api/v1"

    @property
    def session(self) -> requests.Session:
        s = getattr(self._sessions, "session", None)
        if s is None:
            s = _session()
            self._sessions.session = s
        return s

    def _request(self, path: str, params: dict | None = None, absolute: str | None = None) -> dict:
        url = absolute or f"{self.base}/{path.lstrip('/')}"

        headers = {
            "User-Agent": "sd-webui-forge-neo/civitai",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
        except Exception as e:
            raise CivitaiError(f"Could not reach {self.site}: {e}") from e

        if response.status_code == 401:
            raise CivitaiError(f"{self.site} rejected the request (401). Check your API key in Settings -> Neo Optimizations -> Civitai.")
        if response.status_code == 403:
            raise CivitaiError(f"{self.site} refused the request (403). The API key may not have access to this resource.")
        if response.status_code == 404:
            raise CivitaiError(f"Not found on {self.site} (404).")
        if response.status_code == 429:
            raise CivitaiError(f"{self.site} is rate limiting you. Wait a moment and try again.")
        if response.status_code >= 400:
            raise CivitaiError(f"{self.site} returned {response.status_code}: {response.text[:200]}")

        try:
            return response.json()
        except Exception as e:
            raise CivitaiError(f"{self.site} returned something that is not JSON ({response.headers.get('content-type')})") from e

    # ------------------------------------------------------------------ endpoints

    def search(
        self,
        query: str = "",
        types: str | None = None,
        base_models: str | None = None,
        sort: str | None = None,
        period: str = "AllTime",
        nsfw: bool = False,
        limit: int = 24,
        page: int = 1,
    ) -> dict:
        params = {
            "limit": max(1, min(int(limit), 200)),
            "page": max(1, int(page)),
            "nsfw": "true" if nsfw else "false",
            "sort": sort or "Most Downloaded",
        }
        if query:
            params["query"] = query
        if types and types != "Any":
            params["types"] = types
        if base_models and base_models != "Any":
            params["baseModels"] = base_models
        if sort == "Most Downloaded" and period and period != "AllTime":
            params["period"] = period

        return self._request("/models", params=params)

    def get_model(self, model_id) -> dict:
        return self._request(f"/models/{model_id}")

    def get_model_version(self, version_id) -> dict:
        return self._request(f"/model-versions/{version_id}")

    def version_by_hash(self, file_hash: str) -> dict | None:
        """Look a model version up by the SHA256 / AutoV2 hash of a local file."""
        file_hash = (file_hash or "").strip()
        if not file_hash:
            return None
        try:
            return self._request(f"/model-versions/by-hash/{quote(file_hash)}")
        except CivitaiError as e:
            if "404" in str(e):
                return None
            raise

    def creator(self, username: str, limit: int = 24, page: int = 1) -> dict:
        return self._request("/models", params={"username": username, "limit": limit, "page": page, "nsfw": "true"})

    # ------------------------------------------------------------------ helpers

    def download_url(self, version_id, file_name: str | None = None, site: str | None = None) -> str:
        host = f"https://{site or self.site}"
        url = f"{host}/api/download/models/{version_id}"
        if file_name:
            url += f"?{urlencode({'type': 'Model', 'format': 'SafeTensor'})}"
        if self.api_key:
            url += ("&" if "?" in url else "?") + urlencode({"token": self.api_key})
        return url

    def thumbnail(self, url: str, width: int = 320) -> str | None:
        return thumbnail(url, width)

    def fetch_image(self, url: str) -> tuple[bytes, str]:
        """Fetch a preview image (used by the local proxy route)."""
        headers = {"User-Agent": "sd-webui-forge-neo/civitai"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = self.session.get(url, headers=headers, timeout=self.timeout, stream=True)
        except Exception as e:
            raise CivitaiError(str(e)) from e
        if r.status_code >= 400:
            raise CivitaiError(f"image returned {r.status_code}")
        return r.content, r.headers.get("content-type", "image/jpeg")


# --------------------------------------------------------------------------- model helpers


def version_images(model_version: dict, limit: int = 1) -> list[dict]:
    images = model_version.get("images") or []
    images = [i for i in images if isinstance(i, dict) and i.get("url")]
    return images[:limit]


def primary_file(model_version: dict) -> dict | None:
    files = [f for f in (model_version.get("files") or []) if isinstance(f, dict)]

    if not files:
        return None

    for f in files:
        if f.get("primary"):
            return f

    # prefer the safetensors model file
    def score(f):
        s = 0
        if str(f.get("type", "")).lower() == "model":
            s -= 10
        if str(f.get("format", "")).lower() == "safetensor":
            s -= 5
        s += {"Pruned": 0, "ImportError": 50, "Unsafe": 80, "Infected": 100}.get(
            str(f.get("pickleScanResult", "")), 10
        )
        return s

    return sorted(files, key=score)[0]


def file_size_bytes(f: dict) -> int:
    try:
        return int(float(f.get("sizeKB", 0) or 0) * 1024)
    except Exception:
        return 0


def human_size(n: float) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def trained_words(model_version: dict) -> list[str]:
    words = model_version.get("trainedWords") or []
    return [w for w in words if isinstance(w, str) and w.strip()]


def stats_of(model: dict) -> dict:
    s = model.get("stats") or {}
    return {
        "downloads": s.get("downloadCount", 0) or 0,
        "rating": s.get("rating", 0) or 0,
        "rating_count": s.get("ratingCount", 0) or 0,
        "favorites": s.get("favoriteCount", 0) or 0,
    }


def version_label(model_version: dict) -> str:
    name = model_version.get("name") or "unnamed"
    base = model_version.get("baseModel") or "?"
    f = primary_file(model_version)
    size = human_size(file_size_bytes(f or {}))
    return f"{name} - {base} ({size})"


def parse_civitai_url(text: str) -> tuple[int | None, int | None]:
    """Extract ``(model_id, version_id)`` from anything the user pastes."""
    import re

    text = (text or "").strip()
    if not text:
        return None, None

    model_id = version_id = None

    m = re.search(r"models/(\d+)", text)
    if m:
        model_id = int(m.group(1))
    m = re.search(r"modelVersionId=(\d+)", text)
    if m:
        version_id = int(m.group(1))

    if model_id is None and text.isdigit():
        model_id = int(text)

    return model_id, version_id


def rate_limit_sleep(last_call: float, min_interval: float = 0.15) -> float:
    elapsed = time.time() - last_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    return time.time()
