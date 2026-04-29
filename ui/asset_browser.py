"""
Asset Browser panel – left-hand side of the SimReady Browser.

Shows a searchable, filterable grid of asset thumbnails pulled from S3.
Emits asset_selected(AssetInfo) when the user clicks a card.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.s3_client import AssetInfo, S3Client
from styles.nvidia_theme import (
    COLOR_ACCENT,
    COLOR_BG_PANEL,
    COLOR_BG_WIDGET,
    COLOR_BORDER,
    COLOR_TEXT_DISABLED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)

THUMB_SIZE = 128   # px per thumbnail card
DOWNLOAD_THUMBNAILS = True
RENDER_BATCH_SIZE = 24
INITIAL_RENDER_AHEAD_ROWS = 4
LOAD_MORE_ROWS = 8
LOAD_MORE_THRESHOLD_ROWS = 3


class _ThumbnailDecodeSignals(QObject):
    ready = pyqtSignal(str, str, object)  # usd_key, thumbnail path, QImage


class _ThumbnailDecodeWorker(QRunnable):
    """Decode and scale a thumbnail away from the Qt UI thread."""

    def __init__(self, usd_key: str, path: str, size: int, signals: _ThumbnailDecodeSignals):
        super().__init__()
        self.usd_key = usd_key
        self.path = path
        self.size = size
        self.signals = signals
        self.setAutoDelete(True)

    def run(self):
        image = QImage(self.path)
        if not image.isNull():
            image = image.scaled(
                self.size,
                self.size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.signals.ready.emit(self.usd_key, self.path, image)


class AssetBrowserPanel(QWidget):
    """
    Left panel: search bar + category filter + scrollable asset grid.
    """

    asset_selected = pyqtSignal(object)   # AssetInfo
    asset_activated = pyqtSignal(object)  # AssetInfo
    status_message = pyqtSignal(str)

    def __init__(self, s3_client: S3Client, parent=None):
        super().__init__(parent)
        self._client  = s3_client
        self._assets: List[AssetInfo] = []
        self._visible_assets: List[AssetInfo] = []
        self._cards:  List[AssetCard] = []
        self._card_by_usd: Dict[str, AssetCard] = {}
        self._selected: Optional[AssetCard] = None
        self._render_generation = 0
        self._render_index = 0
        self._render_target_index = 0
        self._requested_thumbnails: set[str] = set()
        self._thumbnail_decodes: set[tuple[str, int]] = set()
        self._thumb_decode_signals = _ThumbnailDecodeSignals(self)
        self._thumb_decode_signals.ready.connect(self._on_thumbnail_decoded)
        self._thumb_decode_pool = QThreadPool.globalInstance()
        self._thumb_request_timer = QTimer(self)
        self._thumb_request_timer.setSingleShot(True)
        self._thumb_request_timer.timeout.connect(self._request_visible_thumbnails)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self._filter_assets)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._filter_assets)
        self._last_column_count = 0

        self._build_ui()
        self._connect_signals()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Header ───────────────────────────────────────────────────────────
        header = QLabel("SimReady Assets")
        header.setObjectName("section_header")
        header.setStyleSheet(
            f"color: {COLOR_ACCENT}; font-size: 14px; font-weight: bold; "
            f"border-bottom: 1px solid {COLOR_ACCENT}; padding-bottom: 4px;"
        )
        root.addWidget(header)

        # ── Search bar ────────────────────────────────────────────────────────
        search_row = QHBoxLayout()
        search_row.setSpacing(4)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search assets…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._schedule_filter)
        search_row.addWidget(self._search)

        refresh_btn = QToolButton()
        refresh_btn.setText("⟳")
        refresh_btn.setToolTip("Refresh from S3")
        refresh_btn.clicked.connect(self._refresh)
        refresh_btn.setStyleSheet(
            f"QToolButton {{ border: 1px solid {COLOR_BORDER}; border-radius: 4px; "
            f"padding: 4px 6px; color: {COLOR_ACCENT}; font-size: 14px; }}"
            f"QToolButton:hover {{ background: {COLOR_BG_WIDGET}; }}"
        )
        search_row.addWidget(refresh_btn)
        root.addLayout(search_row)

        # ── Category filter ───────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)

        cat_label = QLabel("Category:")
        cat_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px;")
        filter_row.addWidget(cat_label)

        self._cat_combo = QComboBox()
        self._cat_combo.addItem("All")
        self._cat_combo.currentTextChanged.connect(self._schedule_filter)
        filter_row.addWidget(self._cat_combo, 1)

        root.addLayout(filter_row)

        # ── Asset count ───────────────────────────────────────────────────────
        self._count_label = QLabel("Loading…")
        self._count_label.setStyleSheet(f"color: {COLOR_TEXT_DISABLED}; font-size: 10px;")
        root.addWidget(self._count_label)

        # ── Scrollable grid ───────────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._scroll.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: transparent;")
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(4, 4, 4, 4)

        self._scroll.setWidget(self._grid_widget)
        root.addWidget(self._scroll, 1)

        # ── Status / progress bar ─────────────────────────────────────────────
        self._status = QLabel("Connecting to S3…")
        self._status.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; padding: 2px 0;"
        )
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setRange(0, 0)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: #222; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {COLOR_ACCENT}; border-radius: 2px; }}"
        )
        self._progress.hide()
        root.addWidget(self._progress)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(310)
        self.setMaximumWidth(380)

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _connect_signals(self):
        self._client.assets_loaded.connect(self._on_assets_loaded)
        self._client.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._client.status_message.connect(self._on_status)
        self._client.error_occurred.connect(self._on_error)
        self._client.progress_updated.connect(self._on_progress)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _refresh(self):
        self._status.setText("Refreshing…")
        self._progress.setRange(0, 0)
        self._progress.show()
        self._render_generation += 1
        self._visible_assets = []
        self._render_index = 0
        self._clear_grid()
        self._client.refresh(force_network=True)

    def _on_assets_loaded(self, assets: List[AssetInfo]):
        self._assets = assets
        self._update_category_filter()
        self._filter_assets()
        self._status.setText(f"{len(assets)} assets available")
        self._progress.hide()

    def _on_thumbnail_ready(self, asset: AssetInfo):
        card = self._card_by_usd.get(asset.usd_key)
        if not card or not asset.local_thumbnail:
            return

        if card.set_cached_thumbnail(asset.local_thumbnail):
            return

        decode_key = (str(asset.local_thumbnail), THUMB_SIZE)
        if decode_key in self._thumbnail_decodes:
            return
        self._thumbnail_decodes.add(decode_key)
        self._thumb_decode_pool.start(
            _ThumbnailDecodeWorker(asset.usd_key, str(asset.local_thumbnail), THUMB_SIZE, self._thumb_decode_signals)
        )

    def _on_thumbnail_decoded(self, usd_key: str, path: str, image: QImage):
        self._thumbnail_decodes.discard((path, THUMB_SIZE))
        if image.isNull():
            return
        card = self._card_by_usd.get(usd_key)
        if card:
            card.set_thumbnail_image(path, image)

    def _on_status(self, msg: str):
        self._status.setText(msg)
        self.status_message.emit(msg)

    def _on_progress(self, current: int, total: int):
        if total <= 0:
            self._progress.setRange(0, 0)
            self._progress.show()
            return

        self._progress.setRange(0, total)
        self._progress.setValue(max(0, min(current, total)))
        self._progress.setVisible(current < total)

    def _on_error(self, msg: str):
        self._progress.hide()
        self._status.setText(f"⚠ {msg}")
        self.status_message.emit(f"⚠ {msg}")

    # ── Filtering ──────────────────────────────────────────────────────────────

    def _update_category_filter(self):
        cats = sorted({a.category for a in self._assets})
        current = self._cat_combo.currentText()
        self._cat_combo.blockSignals(True)
        self._cat_combo.clear()
        self._cat_combo.addItem("All")
        for c in cats:
            self._cat_combo.addItem(c)
        # Restore selection if still valid
        idx = self._cat_combo.findText(current)
        self._cat_combo.setCurrentIndex(max(0, idx))
        self._cat_combo.blockSignals(False)

    def _schedule_filter(self):
        self._filter_timer.start(90)

    def _filter_assets(self):
        query = self._search.text().lower()
        cat   = self._cat_combo.currentText()

        self._visible_assets = [
            a for a in self._assets
            if (not query or query in a.display_name.lower()
                          or query in a.category.lower()
                          or any(query in t.lower() for t in a.tags))
            and (cat == "All" or a.category == cat)
        ]

        self._render_generation += 1
        generation = self._render_generation
        self._render_index = 0
        self._render_target_index = 0
        self._requested_thumbnails.clear()
        self._thumbnail_decodes.clear()
        self._filter_timer.stop()
        self._thumb_request_timer.stop()
        self._clear_grid()
        self._scroll.verticalScrollBar().setValue(0)
        self._count_label.setText(
            f"Showing 0 of {len(self._visible_assets)} filtered assets"
        )

        if self._visible_assets:
            self._progress.hide()
            self._extend_render_target(self._initial_render_target())
            QTimer.singleShot(0, lambda: self._render_asset_batch(generation))
        else:
            self._progress.hide()

    def _render_asset_batch(self, generation: int):
        if generation != self._render_generation:
            return

        cols = self._column_count()
        self._last_column_count = cols
        batch_end = min(
            self._render_index + RENDER_BATCH_SIZE,
            self._render_target_index,
            len(self._visible_assets),
        )
        if batch_end <= self._render_index:
            return

        for i in range(self._render_index, batch_end):
            asset = self._visible_assets[i]
            row, col = divmod(i, cols)
            card = AssetCard(asset, size=THUMB_SIZE)
            card.grid_row = row
            card.clicked.connect(self._on_card_clicked)
            card.double_clicked.connect(self._on_card_double_clicked)
            self._grid.addWidget(card, row, col)
            self._cards.append(card)
            self._card_by_usd[asset.usd_key] = card

        self._render_index = batch_end
        self._count_label.setText(
            f"Showing {self._render_index} of {len(self._visible_assets)} filtered assets"
        )
        self._progress.hide()

        if self._render_index < self._render_target_index:
            QTimer.singleShot(1, lambda: self._render_asset_batch(generation))

        self._schedule_visible_thumbnail_requests()

    def _clear_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = []
        self._card_by_usd = {}
        self._selected = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cols = self._column_count()
        if cols != self._last_column_count and self._visible_assets:
            self._last_column_count = cols
            self._resize_timer.start(80)
        self._schedule_visible_thumbnail_requests()

    def _column_count(self) -> int:
        return max(1, (self._scroll.viewport().width() - 16) // (THUMB_SIZE + 10))

    def _row_step(self) -> int:
        return self._grid.verticalSpacing() + THUMB_SIZE + 30

    def _initial_render_target(self) -> int:
        visible_rows = max(1, (self._scroll.viewport().height() // max(1, self._row_step())) + 1)
        return self._column_count() * (visible_rows + INITIAL_RENDER_AHEAD_ROWS)

    def _extend_render_target(self, count: int) -> None:
        self._render_target_index = min(
            len(self._visible_assets),
            max(self._render_target_index, self._render_index + max(1, count)),
        )

    def _on_scroll_changed(self):
        self._debounce_visible_thumbnail_requests()
        self._maybe_render_more_cards()

    def _maybe_render_more_cards(self):
        if self._render_index >= len(self._visible_assets):
            return

        bar = self._scroll.verticalScrollBar()
        threshold = self._row_step() * LOAD_MORE_THRESHOLD_ROWS
        if bar.maximum() - bar.value() > threshold:
            return

        self._extend_render_target(self._column_count() * LOAD_MORE_ROWS)
        QTimer.singleShot(0, lambda gen=self._render_generation: self._render_asset_batch(gen))

    def _schedule_visible_thumbnail_requests(self):
        if not self._thumb_request_timer.isActive():
            self._thumb_request_timer.start(30)

    def _debounce_visible_thumbnail_requests(self):
        self._thumb_request_timer.start(80)

    def _request_visible_thumbnails(self):
        if not DOWNLOAD_THUMBNAILS or not self._cards:
            return

        top = self._scroll.verticalScrollBar().value()
        bottom = top + self._scroll.viewport().height()
        margin = THUMB_SIZE * 2
        row_step = self._grid.verticalSpacing() + THUMB_SIZE + 30

        for card in self._cards:
            card_top = self._grid.contentsMargins().top() + getattr(card, "grid_row", 0) * row_step
            card_bottom = card_top + card.height()
            if card_bottom < top - margin or card_top > bottom + margin:
                continue
            key = card.asset.thumbnail_key or card.asset.usd_key
            if key in self._requested_thumbnails:
                continue
            self._requested_thumbnails.add(key)
            self._client.request_thumbnail(card.asset, allow_download=True)

    # ── Card selection ─────────────────────────────────────────────────────────

    def _on_card_clicked(self, asset: AssetInfo):
        # Deselect previous
        if self._selected:
            self._selected.set_selected(False)

        # Select new
        for card in self._cards:
            if card.asset is asset:
                card.set_selected(True)
                self._selected = card
                break

        self.asset_selected.emit(asset)

    def _on_card_double_clicked(self, asset: AssetInfo):
        self._on_card_clicked(asset)
        self.asset_activated.emit(asset)


# ── Asset card widget ───────────────────────────────────────────────────────────

class AssetCard(QFrame):
    """Single asset thumbnail + name card in the grid."""

    clicked = pyqtSignal(object)  # AssetInfo
    double_clicked = pyqtSignal(object)  # AssetInfo

    _PLACEHOLDER: Optional[QPixmap] = None
    _PIXMAP_CACHE: Dict[tuple[str, int], QPixmap] = {}
    _MAX_PIXMAP_CACHE = 256

    def __init__(self, asset: AssetInfo, size: int = 128, parent=None):
        super().__init__(parent)
        self.asset = asset
        self._size = size
        self.setObjectName("asset_card")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{asset.display_name}\n{asset.category}\n{asset.usd_key}")
        self.setFixedSize(size + 8, size + 30)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setFixedSize(size, size)
        self._thumb.setPixmap(self._get_placeholder(size))
        layout.addWidget(self._thumb)

        name = QLabel(asset.display_name)
        name.setObjectName("asset_name")
        name.setAlignment(Qt.AlignCenter)
        name.setWordWrap(False)
        name.setStyleSheet(
            f"color: {COLOR_TEXT_PRIMARY}; font-size: 10px; background: transparent;"
        )
        # Truncate long names
        fm = name.fontMetrics()
        elided = fm.elidedText(asset.display_name, Qt.ElideRight, size)
        name.setText(elided)
        layout.addWidget(name)

        self._update_style(selected=False)

    # ── Public ─────────────────────────────────────────────────────────────────

    def set_thumbnail(self, path):
        if path is None:
            return
        if self.set_cached_thumbnail(path):
            return
        image = QImage(str(path))
        if not image.isNull():
            image = image.scaled(
                self._size,
                self._size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.set_thumbnail_image(str(path), image)

    def set_cached_thumbnail(self, path) -> bool:
        cache_key = (str(path), self._size)
        cached = self._PIXMAP_CACHE.get(cache_key)
        if cached:
            self._thumb.setPixmap(cached)
            return True
        return False

    def set_thumbnail_image(self, path, image: QImage) -> None:
        if image.isNull():
            return
        try:
            px = QPixmap.fromImage(image)
            if len(self._PIXMAP_CACHE) >= self._MAX_PIXMAP_CACHE:
                self._PIXMAP_CACHE.pop(next(iter(self._PIXMAP_CACHE)))
            self._PIXMAP_CACHE[(str(path), self._size)] = px
            self._thumb.setPixmap(px)
        except Exception:
            pass

    def set_selected(self, selected: bool):
        self._update_style(selected)

    # ── Internal ───────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.asset)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.asset)
            event.accept()

    def _update_style(self, selected: bool):
        if selected:
            self.setStyleSheet(
                f"QFrame#asset_card {{ background: #2a3d1a; "
                f"border: 2px solid {COLOR_ACCENT}; border-radius: 6px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#asset_card {{ background: {COLOR_BG_WIDGET}; "
                f"border: 1px solid {COLOR_BORDER}; border-radius: 6px; }}"
                f"QFrame#asset_card:hover {{ border: 1px solid {COLOR_ACCENT}; "
                f"background: #2c2c2c; }}"
            )

    @classmethod
    def _get_placeholder(cls, size: int) -> QPixmap:
        if cls._PLACEHOLDER and cls._PLACEHOLDER.width() == size:
            return cls._PLACEHOLDER
        px = QPixmap(size, size)
        px.fill(QColor("#1e1e1e"))
        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)
        # Draw USD icon placeholder
        p.setPen(QPen(QColor("#3a3a3a"), 2))
        c = size // 2
        r = size // 3
        p.drawEllipse(c - r, c - r, r * 2, r * 2)
        p.setPen(QColor("#4a4a4a"))
        p.setFont(QFont("Segoe UI", size // 8))
        p.drawText(px.rect(), Qt.AlignCenter, "USD")
        p.end()
        cls._PLACEHOLDER = px
        return px
