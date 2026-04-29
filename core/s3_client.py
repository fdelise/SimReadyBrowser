"""Public S3 browser for NVIDIA Isaac SimReady assets."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional
from urllib.error import URLError

from PyQt5.QtCore import QObject, QRunnable, QThread, QThreadPool, pyqtSignal

S3_BUCKET = "omniverse-content-production"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com"
S3_BUCKET_PATH = "Assets/Isaac/6.0/Isaac/SimReady/"
S3_URI_ROOT = f"s3://{S3_BUCKET}/{S3_BUCKET_PATH}"

MANIFEST_NAMES = ["manifest.json", "index.json", "catalog.json", "assets.json"]
THUMBNAIL_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
USD_EXTS = [".usd", ".usda", ".usdc", ".usdz"]
CATALOG_CACHE_NAME = "simready_asset_catalog.json"
WORKSPACE_CACHE_PATH = Path(__file__).resolve().parents[1] / "cache" / CATALOG_CACHE_NAME


@dataclass
class AssetInfo:
    """A single top-level SimReady USD asset."""

    name: str
    usd_key: str
    thumbnail_key: Optional[str] = None
    category: str = "General"
    tags: list[str] = field(default_factory=list)
    description: str = ""
    local_usd: Optional[Path] = None
    local_thumbnail: Optional[Path] = None

    @property
    def usd_url(self) -> str:
        return f"{S3_BASE_URL}/{urllib.parse.quote(self.usd_key, safe='/')}"

    @property
    def s3_uri(self) -> str:
        return f"s3://{S3_BUCKET}/{self.usd_key}"

    @property
    def thumbnail_url(self) -> Optional[str]:
        if not self.thumbnail_key:
            return None
        return f"{S3_BASE_URL}/{urllib.parse.quote(self.thumbnail_key, safe='/')}"

    @property
    def display_name(self) -> str:
        return self.name or PurePosixPath(self.usd_key).stem.replace("_", " ").title()


class WorkerSignals(QObject):
    progress = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()


class DownloadWorker(QRunnable):
    """Downloads one public S3 URL to a local cache path."""

    def __init__(self, url: str, dest: Path, signals: WorkerSignals, user_data=None, timeout_s: float = 15.0):
        super().__init__()
        self.url = url
        self.dest = dest
        self.signals = signals
        self.user_data = user_data
        self.timeout_s = timeout_s
        self.setAutoDelete(True)

    def run(self):
        try:
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(self.url, headers={"User-Agent": "SimReadyBrowser/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                self.dest.write_bytes(resp.read())
            self.signals.result.emit({"path": self.dest, "user_data": self.user_data})
        except Exception as exc:
            self.signals.error.emit(f"Download failed {self.url}: {exc}")
        finally:
            self.signals.finished.emit()


class DiscoveryThread(QThread):
    """Runs blocking S3 enumeration away from the UI thread."""

    result = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, client: "S3Client"):
        super().__init__(client)
        self._client = client

    def run(self):
        try:
            assets = self._client._try_manifest()
            if self.isInterruptionRequested():
                return
            if not assets:
                self._client.status_message.emit("No manifest found - enumerating bucket...")
                assets = self._client._enumerate_bucket()
            if not self.isInterruptionRequested():
                self.result.emit(assets)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error.emit(f"Discovery error: {exc}")


class S3Client(QObject):
    """Async public S3 catalog and thumbnail client."""

    assets_loaded = pyqtSignal(list)
    thumbnail_ready = pyqtSignal(object)
    usd_ready = pyqtSignal(object)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)

    def __init__(self, cache_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.cache_dir = cache_dir or Path.home() / ".cache" / "simready_browser"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(6)
        self._threads: list[DiscoveryThread] = []
        self._assets: list[AssetInfo] = []
        self._thumbnail_requests: set[str] = set()
        self._shutting_down = False

    def refresh(self, force_network: bool = False) -> None:
        """Discover all top-level SimReady assets in the public bucket."""
        if self._shutting_down:
            return
        if any(thread.isRunning() for thread in self._threads):
            self.status_message.emit("S3 refresh already in progress...")
            return

        cached_assets = self._load_cached_catalog()
        if cached_assets:
            self._assets = cached_assets
            self.assets_loaded.emit(cached_assets)
            if not force_network:
                self.status_message.emit(f"Loaded {len(cached_assets)} cached assets. Use Refresh to rescan S3.")
                self.progress_updated.emit(len(cached_assets), len(cached_assets))
                return
            self.status_message.emit(f"Loaded {len(cached_assets)} cached assets. Refreshing S3 catalog...")
        else:
            self.status_message.emit("Connecting to S3 bucket...")

        self.progress_updated.emit(0, 0)
        thread = DiscoveryThread(self)
        self._threads.append(thread)
        thread.result.connect(self._on_discovery_done)
        thread.error.connect(self.error_occurred)
        thread.finished.connect(lambda: self._on_thread_finished(thread))
        thread.start()

    def shutdown(self, timeout_ms: int = 15000) -> None:
        """Stop discovery threads before Qt tears down the client."""
        self._shutting_down = True
        for thread in list(self._threads):
            if thread.isRunning():
                thread.requestInterruption()

        for thread in list(self._threads):
            if thread.isRunning():
                thread.quit()
                if not thread.wait(timeout_ms):
                    self.status_message.emit("S3 refresh is still winding down.")
        self._threads = [thread for thread in self._threads if thread.isRunning()]
        self._thumbnail_requests.clear()
        try:
            self._pool.clear()
            if not self._pool.waitForDone(timeout_ms):
                self.status_message.emit("Thumbnail downloads are still winding down.")
        except Exception:
            pass

    def request_thumbnail(self, asset: AssetInfo, allow_download: bool = True) -> None:
        if self._shutting_down:
            return
        if asset.local_thumbnail and asset.local_thumbnail.exists():
            self.thumbnail_ready.emit(asset)
            return
        if not asset.thumbnail_url:
            return

        dest = self.cache_dir / "thumbnails" / _safe_filename(asset.thumbnail_key or "")
        if dest.exists():
            asset.local_thumbnail = dest
            self.thumbnail_ready.emit(asset)
            return

        if not allow_download:
            return
        request_key = asset.thumbnail_key or asset.thumbnail_url
        if request_key in self._thumbnail_requests:
            return
        self._thumbnail_requests.add(request_key)

        sig = WorkerSignals()
        sig.result.connect(lambda result, key=request_key: self._on_thumbnail_done(asset, result, key))
        sig.error.connect(lambda msg, key=request_key: self._on_thumbnail_error(key, msg))
        self._pool.start(DownloadWorker(asset.thumbnail_url, dest, sig, asset, timeout_s=8.0))

    def request_usd(self, asset: AssetInfo) -> None:
        """Optional top-level USD cache. The viewport normally streams usd_url."""
        if self._shutting_down:
            return
        if asset.local_usd and asset.local_usd.exists():
            self.usd_ready.emit(asset)
            return

        dest = self.cache_dir / "usd" / _safe_filename(asset.usd_key)
        if dest.exists():
            asset.local_usd = dest
            self.usd_ready.emit(asset)
            return

        self.status_message.emit(f"Downloading {asset.display_name}...")
        sig = WorkerSignals()
        sig.result.connect(lambda result: self._on_usd_done(asset, result))
        sig.error.connect(self.error_occurred)
        self._pool.start(DownloadWorker(asset.usd_url, dest, sig, asset, timeout_s=30.0))

    @property
    def assets(self) -> list[AssetInfo]:
        return list(self._assets)

    def _on_discovery_done(self, assets: list[AssetInfo]) -> None:
        if self._shutting_down:
            return
        if not assets:
            self.progress_updated.emit(0, 1)
            return
        self._assets = assets
        self._save_cached_catalog(assets)
        self.status_message.emit(f"Found {len(assets)} assets.")
        self.progress_updated.emit(len(assets), len(assets))
        self.assets_loaded.emit(assets)

    def _load_cached_catalog(self) -> list[AssetInfo]:
        for path in self._catalog_cache_paths():
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("bucket") != S3_BUCKET or data.get("prefix") != S3_BUCKET_PATH:
                    continue
                assets = [self._asset_from_cache(item) for item in data.get("assets", [])]
                assets = [asset for asset in assets if asset is not None]
                if assets:
                    return assets
            except Exception:
                continue
        return []

    def _save_cached_catalog(self, assets: list[AssetInfo]) -> None:
        payload = {
            "bucket": S3_BUCKET,
            "prefix": S3_BUCKET_PATH,
            "asset_count": len(assets),
            "assets": [self._asset_to_cache(asset) for asset in assets],
        }
        for path in self._catalog_cache_paths():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception as exc:
                self.status_message.emit(f"Catalog cache write skipped: {exc}")

    def _catalog_cache_paths(self) -> list[Path]:
        paths = [self.cache_dir / CATALOG_CACHE_NAME, WORKSPACE_CACHE_PATH]
        unique_paths: list[Path] = []
        for path in paths:
            if path not in unique_paths:
                unique_paths.append(path)
        return unique_paths

    @staticmethod
    def _asset_to_cache(asset: AssetInfo) -> dict:
        return {
            "name": asset.name,
            "usd_key": asset.usd_key,
            "thumbnail_key": asset.thumbnail_key,
            "category": asset.category,
            "tags": asset.tags,
            "description": asset.description,
        }

    @staticmethod
    def _asset_from_cache(item: dict) -> Optional[AssetInfo]:
        if not isinstance(item, dict) or not item.get("usd_key"):
            return None
        return AssetInfo(
            name=str(item.get("name", "")),
            usd_key=str(item["usd_key"]),
            thumbnail_key=item.get("thumbnail_key"),
            category=str(item.get("category", "General")),
            tags=list(item.get("tags", [])),
            description=str(item.get("description", "")),
        )

    def _try_manifest(self) -> list[AssetInfo]:
        for name in MANIFEST_NAMES:
            if _discovery_interrupted():
                return []
            url = f"{S3_BASE_URL}/{S3_BUCKET_PATH}{name}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SimReadyBrowser/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.status_message.emit(f"Loaded manifest: {name}")
                return self._parse_manifest(data)
            except Exception:
                continue
        return []

    def _parse_manifest(self, data) -> list[AssetInfo]:
        raw_list = data
        if isinstance(data, dict):
            raw_list = data.get("assets", data.get("items", []))
        if not isinstance(raw_list, list):
            return []

        assets: list[AssetInfo] = []
        for item in raw_list:
            if isinstance(item, str):
                item = {"path": item}
            if not isinstance(item, dict):
                continue

            usd_key = item.get("path", item.get("usd", item.get("url", "")))
            if not usd_key:
                continue
            usd_key = str(usd_key).lstrip("/")
            if not usd_key.startswith("Assets/"):
                usd_key = f"{S3_BUCKET_PATH}{usd_key}"

            stem = re.sub(r"\.(usd[azc]?|usdz)$", "", usd_key, flags=re.IGNORECASE)
            thumb_key = item.get("thumbnail", item.get("preview", f"{stem}.png"))
            if thumb_key and not str(thumb_key).startswith("Assets/"):
                thumb_key = f"{S3_BUCKET_PATH}{str(thumb_key).lstrip('/')}"

            assets.append(
                AssetInfo(
                    name=item.get("name", item.get("label", "")),
                    usd_key=usd_key,
                    thumbnail_key=thumb_key,
                    category=item.get("category", item.get("group", "General")),
                    tags=item.get("tags", item.get("keywords", [])),
                    description=item.get("description", item.get("desc", "")),
                )
            )
        return assets

    def _enumerate_bucket(self) -> list[AssetInfo]:
        all_keys: list[str] = []
        next_token: Optional[str] = None

        while True:
            if _discovery_interrupted():
                return []
            keys, next_token = self._list_page(S3_BUCKET_PATH, next_token)
            all_keys.extend(keys)
            self.status_message.emit(f"Scanning bucket... {len(all_keys)} objects found")
            self.progress_updated.emit(len(all_keys), 0)
            if not next_token:
                break

        usd_keys = {key for key in all_keys if _is_main_asset_usd(key)}
        thumb_keys = {key for key in all_keys if PurePosixPath(key).suffix.lower() in THUMBNAIL_EXTS}

        assets: list[AssetInfo] = []
        for usd_key in sorted(usd_keys):
            parent_dir = str(PurePosixPath(usd_key).parent)
            usd_name = PurePosixPath(usd_key).name
            usd_stem = PurePosixPath(usd_key).stem
            stem = re.sub(r"\.(usd[azc]?|usdz)$", "", usd_key, flags=re.IGNORECASE)

            thumbnail_key = self._choose_thumbnail(parent_dir, usd_name, usd_stem, stem, thumb_keys)
            rel = _relative_asset_key(usd_key)
            parts = PurePosixPath(rel).parts
            category = " / ".join(parts[:3]) if len(parts) >= 3 else (parts[0] if parts else "General")
            tags = [part.replace("_", " ") for part in parts[:-1]]

            assets.append(
                AssetInfo(
                    name=usd_stem.replace("_", " ").title(),
                    usd_key=usd_key,
                    thumbnail_key=thumbnail_key,
                    category=category,
                    tags=tags,
                )
            )
        return assets

    def _choose_thumbnail(
        self,
        parent_dir: str,
        usd_name: str,
        usd_stem: str,
        stem: str,
        thumb_keys: set[str],
    ) -> Optional[str]:
        candidates: list[str] = []
        for size in ("512x512", "256x256", "128x128"):
            candidates.append(f"{parent_dir}/.thumbs/{size}/{usd_name}.png")
            candidates.append(f"{parent_dir}/.thumbs/{size}/{usd_name}.auto.png")
        candidates.append(f"{parent_dir}/.thumbs/{usd_stem}_thumbnail.png")
        candidates.append(f"{parent_dir}/.thumbs/{usd_stem}.png")

        for ext in THUMBNAIL_EXTS:
            candidates.append(f"{stem}{ext}")
            for tname in ("thumbnail", "preview", "icon"):
                candidates.append(f"{parent_dir}/{tname}{ext}")

        for candidate in candidates:
            if candidate in thumb_keys:
                return candidate
        return None

    def _list_page(self, prefix: str, continuation_token: Optional[str] = None) -> tuple[list[str], Optional[str]]:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        url = f"{S3_BASE_URL}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SimReadyBrowser/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
        except URLError as exc:
            raise RuntimeError(f"S3 list request failed: {exc}") from exc

        root = ET.fromstring(xml_data)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys = [
            node.findtext("s3:Key", namespaces=ns)
            for node in root.findall("s3:Contents", ns)
            if node.findtext("s3:Key", namespaces=ns)
        ]
        next_tok_el = root.find("s3:NextContinuationToken", ns)
        next_token = next_tok_el.text if next_tok_el is not None else None
        return keys, next_token

    def _on_thumbnail_done(self, asset: AssetInfo, result: dict, request_key: str) -> None:
        self._thumbnail_requests.discard(request_key)
        if self._shutting_down:
            return
        asset.local_thumbnail = result["path"]
        self.thumbnail_ready.emit(asset)

    def _on_thumbnail_error(self, request_key: str, msg: str) -> None:
        self._thumbnail_requests.discard(request_key)
        if self._shutting_down:
            return
        self.error_occurred.emit(msg)

    def _on_usd_done(self, asset: AssetInfo, result: dict) -> None:
        if self._shutting_down:
            return
        asset.local_usd = result["path"]
        self.usd_ready.emit(asset)

    def _on_thread_finished(self, thread: DiscoveryThread) -> None:
        if thread in self._threads:
            self._threads.remove(thread)
        thread.deleteLater()


def _safe_filename(key: str) -> str:
    return key.replace("//", "/").lstrip("/")


def _discovery_interrupted() -> bool:
    return QThread.currentThread().isInterruptionRequested()


def _relative_asset_key(key: str) -> str:
    if key.startswith(S3_BUCKET_PATH):
        return key[len(S3_BUCKET_PATH):]
    return key


def _is_main_asset_usd(key: str) -> bool:
    if PurePosixPath(key).suffix.lower() not in USD_EXTS:
        return False
    rel_lower = f"/{_relative_asset_key(key).lower()}"
    ignored_parts = ("/payloads/", "/textures/", "/.thumbs/")
    return not any(part in rel_lower for part in ignored_parts)
