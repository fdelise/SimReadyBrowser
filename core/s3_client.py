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
S3_SOURCE_NAME = "Isaac SimReady"

MANIFEST_NAMES = ["manifest.json", "index.json", "catalog.json", "assets.json"]
THUMBNAIL_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
USD_EXTS = [".usd", ".usda", ".usdc", ".usdz"]
CATALOG_CACHE_NAME = "simready_asset_catalog.json"
S3_LOCATIONS_CACHE_NAME = "simready_s3_locations.json"
WORKSPACE_CACHE_PATH = Path(__file__).resolve().parents[1] / "cache" / CATALOG_CACHE_NAME


def _normalize_s3_prefix(prefix: str) -> str:
    text = str(prefix or "").strip().lstrip("/")
    if text and not text.endswith("/"):
        text += "/"
    return text


def _default_source_name(bucket: str, prefix: str) -> str:
    if bucket == S3_BUCKET and _normalize_s3_prefix(prefix) == S3_BUCKET_PATH:
        return S3_SOURCE_NAME
    parts = [part for part in PurePosixPath(_normalize_s3_prefix(prefix)).parts if part]
    if parts:
        return f"{bucket} / {' / '.join(parts[-2:])}"
    return bucket


@dataclass(frozen=True)
class S3Location:
    """A public S3 bucket/prefix that can be scanned for USD assets."""

    bucket: str
    prefix: str = ""
    base_url: str = ""
    name: str = ""

    def __post_init__(self):
        bucket = str(self.bucket or "").strip()
        prefix = _normalize_s3_prefix(self.prefix)
        default_base_url = S3_BASE_URL if bucket == S3_BUCKET else f"https://{bucket}.s3.amazonaws.com"
        base_url = str(self.base_url or default_base_url).rstrip("/")
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "name", str(self.name or _default_source_name(bucket, prefix)))

    @property
    def root_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}" if self.prefix else f"s3://{self.bucket}/"

    @property
    def display_name(self) -> str:
        return self.name or self.root_uri

    def object_url(self, key: str) -> str:
        return f"{self.base_url}/{urllib.parse.quote(str(key).lstrip('/'), safe='/')}"

    def to_cache(self) -> dict:
        return {
            "bucket": self.bucket,
            "prefix": self.prefix,
            "base_url": self.base_url,
            "name": self.name,
        }


DEFAULT_S3_LOCATION = S3Location(
    bucket=S3_BUCKET,
    prefix=S3_BUCKET_PATH,
    base_url=S3_BASE_URL,
    name=S3_SOURCE_NAME,
)


@dataclass
class AssetInfo:
    """A single top-level SimReady USD asset."""

    name: str
    usd_key: str
    thumbnail_key: Optional[str] = None
    category: str = "General"
    tags: list[str] = field(default_factory=list)
    description: str = ""
    bucket: str = S3_BUCKET
    base_url: str = S3_BASE_URL
    source_prefix: str = S3_BUCKET_PATH
    source_uri: str = S3_URI_ROOT
    source_name: str = S3_SOURCE_NAME
    local_usd: Optional[Path] = None
    local_thumbnail: Optional[Path] = None

    @property
    def usd_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{urllib.parse.quote(self.usd_key, safe='/')}"

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.usd_key}"

    @property
    def asset_id(self) -> str:
        return f"{self.bucket}/{self.usd_key}"

    @property
    def thumbnail_url(self) -> Optional[str]:
        if not self.thumbnail_key:
            return None
        return f"{self.base_url.rstrip('/')}/{urllib.parse.quote(self.thumbnail_key, safe='/')}"

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
        all_assets: list[AssetInfo] = []
        errors: list[str] = []
        locations = self._client.locations
        for location in locations:
            if self.isInterruptionRequested():
                return
            try:
                self._client.status_message.emit(f"Scanning {location.display_name}...")
                assets = self._client._try_manifest(location)
                if self.isInterruptionRequested():
                    return
                if not assets:
                    self._client.status_message.emit(
                        f"No manifest found for {location.display_name} - enumerating bucket..."
                    )
                    assets = self._client._enumerate_bucket(location)
                all_assets.extend(assets)
            except Exception as exc:
                errors.append(f"{location.root_uri}: {exc}")

        if self.isInterruptionRequested():
            return
        if all_assets:
            if errors:
                self._client.error_occurred.emit("Some S3 locations failed: " + "; ".join(errors))
            self.result.emit(all_assets)
        elif errors:
            self.error.emit("Discovery error: " + "; ".join(errors))
        else:
            self.result.emit([])


class S3Client(QObject):
    """Async public S3 catalog and thumbnail client."""

    assets_loaded = pyqtSignal(list)
    thumbnail_ready = pyqtSignal(object)
    usd_ready = pyqtSignal(object)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)
    locations_changed = pyqtSignal(list)

    def __init__(self, cache_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.cache_dir = cache_dir or Path.home() / ".cache" / "simready_browser"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(6)
        self._threads: list[DiscoveryThread] = []
        self._assets: list[AssetInfo] = []
        self._locations: list[S3Location] = self._load_locations()
        self._thumbnail_requests: set[str] = set()
        self._shutting_down = False

    def refresh(self, force_network: bool = False) -> None:
        """Discover all top-level SimReady assets in the configured S3 locations."""
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

        dest = self.cache_dir / "thumbnails" / _safe_filename(f"{asset.bucket}/{asset.thumbnail_key or ''}")
        if dest.exists():
            asset.local_thumbnail = dest
            self.thumbnail_ready.emit(asset)
            return

        if not allow_download:
            return
        request_key = asset.asset_id + "::thumb::" + str(asset.thumbnail_key or asset.thumbnail_url)
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

        dest = self.cache_dir / "usd" / _safe_filename(asset.asset_id)
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

    @property
    def locations(self) -> list[S3Location]:
        return list(self._locations)

    def add_location(self, uri: str) -> bool:
        location = parse_s3_location(uri)
        existing = {item.root_uri.rstrip("/").lower() for item in self._locations}
        if location.root_uri.rstrip("/").lower() in existing:
            self.status_message.emit(f"S3 location already added: {location.root_uri}")
            return False
        self._locations.append(location)
        self._save_locations()
        self.locations_changed.emit(self.locations)
        self.status_message.emit(f"Added S3 location: {location.root_uri}")
        return True

    def remove_location(self, root_uri: str) -> bool:
        key = str(root_uri or "").rstrip("/").lower()
        if len(self._locations) <= 1:
            self.status_message.emit("Keep at least one S3 location in the asset browser.")
            return False
        kept = [item for item in self._locations if item.root_uri.rstrip("/").lower() != key]
        if len(kept) == len(self._locations):
            return False
        self._locations = kept
        self._assets = [asset for asset in self._assets if asset.source_uri.rstrip("/").lower() != key]
        self._save_locations()
        self.locations_changed.emit(self.locations)
        self.assets_loaded.emit(self.assets)
        self.status_message.emit(f"Removed S3 location: {root_uri}")
        return True

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
                if not self._cached_catalog_matches_locations(data):
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
            "schema": 2,
            "locations": [location.to_cache() for location in self._locations],
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

    def _locations_cache_path(self) -> Path:
        return self.cache_dir / S3_LOCATIONS_CACHE_NAME

    def _load_locations(self) -> list[S3Location]:
        path = self._locations_cache_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                raw_locations = data.get("locations", data if isinstance(data, list) else [])
                locations = [_location_from_cache(item) for item in raw_locations]
                locations = [location for location in locations if location is not None]
                if locations:
                    return _dedupe_locations(locations)
            except Exception:
                pass
        return [DEFAULT_S3_LOCATION]

    def _save_locations(self) -> None:
        payload = {"locations": [location.to_cache() for location in self._locations]}
        try:
            path = self._locations_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self.status_message.emit(f"S3 location cache write skipped: {exc}")

    def _cached_catalog_matches_locations(self, data: dict) -> bool:
        raw_locations = data.get("locations")
        if raw_locations is None:
            legacy_location = S3Location(
                bucket=str(data.get("bucket", "")),
                prefix=str(data.get("prefix", "")),
                base_url=S3_BASE_URL,
                name=S3_SOURCE_NAME,
            )
            return _location_keys([legacy_location]) == _location_keys(self._locations)

        locations = [_location_from_cache(item) for item in raw_locations]
        locations = [location for location in locations if location is not None]
        return _location_keys(locations) == _location_keys(self._locations)

    @staticmethod
    def _asset_to_cache(asset: AssetInfo) -> dict:
        return {
            "name": asset.name,
            "usd_key": asset.usd_key,
            "thumbnail_key": asset.thumbnail_key,
            "category": asset.category,
            "tags": asset.tags,
            "description": asset.description,
            "bucket": asset.bucket,
            "base_url": asset.base_url,
            "source_prefix": asset.source_prefix,
            "source_uri": asset.source_uri,
            "source_name": asset.source_name,
        }

    @staticmethod
    def _asset_from_cache(item: dict) -> Optional[AssetInfo]:
        if not isinstance(item, dict) or not item.get("usd_key"):
            return None
        usd_key = str(item["usd_key"])
        bucket = str(item.get("bucket", S3_BUCKET))
        source_prefix = _normalize_s3_prefix(item.get("source_prefix", S3_BUCKET_PATH))
        source_uri = str(item.get("source_uri") or f"s3://{bucket}/{source_prefix}")
        base_url = str(item.get("base_url") or (S3_BASE_URL if bucket == S3_BUCKET else f"https://{bucket}.s3.amazonaws.com"))
        source_name = str(item.get("source_name") or _default_source_name(bucket, source_prefix))
        return AssetInfo(
            name=str(item.get("name", "")),
            usd_key=usd_key,
            thumbnail_key=item.get("thumbnail_key"),
            category=_category_from_asset_key(usd_key, str(item.get("category", "General")), source_prefix),
            tags=list(item.get("tags", [])),
            description=str(item.get("description", "")),
            bucket=bucket,
            base_url=base_url,
            source_prefix=source_prefix,
            source_uri=source_uri,
            source_name=source_name,
        )

    def _try_manifest(self, location: S3Location) -> list[AssetInfo]:
        for name in MANIFEST_NAMES:
            if _discovery_interrupted():
                return []
            url = location.object_url(f"{location.prefix}{name}")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SimReadyBrowser/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.status_message.emit(f"Loaded manifest for {location.display_name}: {name}")
                return self._parse_manifest(data, location)
            except Exception:
                continue
        return []

    def _parse_manifest(self, data, location: S3Location = DEFAULT_S3_LOCATION) -> list[AssetInfo]:
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
            usd_key = _resolve_location_key(str(usd_key), location)

            stem = re.sub(r"\.(usd[azc]?|usdz)$", "", usd_key, flags=re.IGNORECASE)
            thumb_key = item.get("thumbnail", item.get("preview", f"{stem}.png"))
            if thumb_key:
                thumb_key = _resolve_location_key(str(thumb_key), location)

            assets.append(
                AssetInfo(
                    name=item.get("name", item.get("label", "")),
                    usd_key=usd_key,
                    thumbnail_key=thumb_key,
                    category=_category_from_asset_key(
                        usd_key,
                        item.get("category", item.get("group", "General")),
                        location.prefix,
                    ),
                    tags=item.get("tags", item.get("keywords", [])),
                    description=item.get("description", item.get("desc", "")),
                    bucket=location.bucket,
                    base_url=location.base_url,
                    source_prefix=location.prefix,
                    source_uri=location.root_uri,
                    source_name=location.display_name,
                )
            )
        return assets

    def _enumerate_bucket(self, location: S3Location = DEFAULT_S3_LOCATION) -> list[AssetInfo]:
        all_keys: list[str] = []
        next_token: Optional[str] = None

        while True:
            if _discovery_interrupted():
                return []
            keys, next_token = self._list_page(location, next_token)
            all_keys.extend(keys)
            self.status_message.emit(f"Scanning {location.display_name}... {len(all_keys)} objects found")
            self.progress_updated.emit(len(all_keys), 0)
            if not next_token:
                break

        usd_keys = {key for key in all_keys if _is_main_asset_usd(key, location.prefix)}
        thumb_keys = {key for key in all_keys if PurePosixPath(key).suffix.lower() in THUMBNAIL_EXTS}

        assets: list[AssetInfo] = []
        for usd_key in sorted(usd_keys):
            parent_dir = str(PurePosixPath(usd_key).parent)
            usd_name = PurePosixPath(usd_key).name
            usd_stem = PurePosixPath(usd_key).stem
            stem = re.sub(r"\.(usd[azc]?|usdz)$", "", usd_key, flags=re.IGNORECASE)

            thumbnail_key = self._choose_thumbnail(parent_dir, usd_name, usd_stem, stem, thumb_keys)
            rel = _relative_asset_key(usd_key, location.prefix)
            parts = PurePosixPath(rel).parts
            category = _category_from_asset_key(usd_key, source_prefix=location.prefix)
            tags = [part.replace("_", " ") for part in parts[:-1]]

            assets.append(
                AssetInfo(
                    name=usd_stem.replace("_", " ").title(),
                    usd_key=usd_key,
                    thumbnail_key=thumbnail_key,
                    category=category,
                    tags=tags,
                    bucket=location.bucket,
                    base_url=location.base_url,
                    source_prefix=location.prefix,
                    source_uri=location.root_uri,
                    source_name=location.display_name,
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

    def _list_page(self, location: S3Location, continuation_token: Optional[str] = None) -> tuple[list[str], Optional[str]]:
        params = {
            "list-type": "2",
            "prefix": location.prefix,
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        url = f"{location.base_url}?{urllib.parse.urlencode(params)}"
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


def parse_s3_location(uri: str) -> S3Location:
    """Parse an s3:// or public S3 HTTPS location into a scannable bucket/prefix."""
    text = str(uri or "").strip()
    if not text:
        raise ValueError("Enter an S3 location, for example s3://bucket/path/.")

    parsed = urllib.parse.urlparse(text)
    if parsed.scheme.lower() == "s3":
        bucket = parsed.netloc.strip()
        prefix = parsed.path.lstrip("/")
        if not bucket:
            raise ValueError("S3 location is missing a bucket name.")
        return S3Location(bucket=bucket, prefix=prefix)

    if parsed.scheme.lower() in {"http", "https"}:
        host = parsed.netloc.strip()
        path_parts = [part for part in parsed.path.split("/") if part]
        if not host:
            raise ValueError("S3 URL is missing a host.")

        host_parts = host.split(".")
        if len(host_parts) >= 3 and host_parts[1] == "s3":
            bucket = host_parts[0]
            prefix = "/".join(path_parts)
            return S3Location(
                bucket=bucket,
                prefix=prefix,
                base_url=f"{parsed.scheme}://{host}",
            )

        if host.startswith("s3.") and path_parts:
            bucket = path_parts[0]
            prefix = "/".join(path_parts[1:])
            return S3Location(
                bucket=bucket,
                prefix=prefix,
                base_url=f"{parsed.scheme}://{host}/{bucket}",
            )

    raise ValueError("Use an s3://bucket/path or public S3 https:// URL.")


def _location_from_cache(item) -> Optional[S3Location]:
    try:
        if isinstance(item, str):
            return parse_s3_location(item)
        if not isinstance(item, dict):
            return None
        bucket = str(item.get("bucket", "")).strip()
        if not bucket:
            return None
        return S3Location(
            bucket=bucket,
            prefix=str(item.get("prefix", "")),
            base_url=str(item.get("base_url", "")),
            name=str(item.get("name", "")),
        )
    except Exception:
        return None


def _dedupe_locations(locations: list[S3Location]) -> list[S3Location]:
    seen: set[str] = set()
    result: list[S3Location] = []
    for location in locations:
        key = location.root_uri.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(location)
    return result


def _location_keys(locations: list[S3Location]) -> list[str]:
    return sorted(location.root_uri.rstrip("/").lower() for location in locations)


def _resolve_location_key(value: str, location: S3Location) -> str:
    text = str(value or "").strip()
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme.lower() == "s3":
        text = parsed.path.lstrip("/")
    elif parsed.scheme.lower() in {"http", "https"}:
        text = parsed.path.lstrip("/")
        if text.startswith(f"{location.bucket}/"):
            text = text[len(location.bucket) + 1 :]
    else:
        text = text.lstrip("/")

    if not text:
        return location.prefix
    if text.startswith(location.prefix) or text.startswith("Assets/"):
        return text
    return f"{location.prefix}{text}"


def _relative_asset_key(key: str, source_prefix: str = S3_BUCKET_PATH) -> str:
    prefix = _normalize_s3_prefix(source_prefix)
    if prefix and key.startswith(prefix):
        return key[len(prefix):]
    if key.startswith(S3_BUCKET_PATH):
        return key[len(S3_BUCKET_PATH):]
    return key


def _category_from_asset_key(key: str, fallback: str = "General", source_prefix: str = S3_BUCKET_PATH) -> str:
    """Return the visible browser category, hiding the asset leaf folder."""
    rel = _relative_asset_key(str(key or "").strip(), source_prefix)
    parts = [part for part in PurePosixPath(rel).parts if part and part not in {".", "/"}]
    if not parts:
        return str(fallback or "General")

    dirs = list(parts[:-1])
    if len(dirs) >= 2:
        parent_dirs = dirs[:-1]
        if parent_dirs:
            return " / ".join(parent_dirs)
    if dirs:
        return " / ".join(dirs)
    return str(fallback or "General")


def _is_main_asset_usd(key: str, source_prefix: str = S3_BUCKET_PATH) -> bool:
    if PurePosixPath(key).suffix.lower() not in USD_EXTS:
        return False
    rel_lower = f"/{_relative_asset_key(key, source_prefix).lower()}"
    ignored_parts = ("/payloads/", "/textures/", "/.thumbs/")
    return not any(part in rel_lower for part in ignored_parts)
