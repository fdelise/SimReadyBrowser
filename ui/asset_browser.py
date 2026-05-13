"""
Asset Browser panel – left-hand side of the SimReady Browser.

Shows a searchable, filterable grid of asset thumbnails pulled from S3.
Emits asset_selected(AssetInfo) when the user clicks a card, and
load_selected_requested(list[AssetInfo]) when the user loads a multi-selection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

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
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.s3_client import AssetInfo, S3Client, parse_s3_location
from styles.nvidia_theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_DIM,
    COLOR_ACCENT_HOVER,
    COLOR_BG_HOVER,
    COLOR_BG_PANEL,
    COLOR_BG_WIDGET,
    COLOR_BORDER,
    COLOR_TEXT_DISABLED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
)

THUMB_SIZE = 88   # px per thumbnail image
CARD_WIDTH_PAD = 10
CARD_HEIGHT_PAD = 34
CARD_GRID_GAP = 6
DOWNLOAD_THUMBNAILS = True
RENDER_BATCH_SIZE = 48
INITIAL_RENDER_AHEAD_ROWS = 5
LOAD_MORE_ROWS = 10
LOAD_MORE_THRESHOLD_ROWS = 3
BROWSER_MIN_WIDTH = 310
BROWSER_MAX_WIDTH = 380


class _ThumbnailDecodeSignals(QObject):
    ready = pyqtSignal(str, str, int, object)  # usd_key, thumbnail path, size, QImage


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
        self.signals.ready.emit(self.usd_key, self.path, self.size, image)


class AssetBrowserPanel(QWidget):
    """
    Left panel: search bar + category filter + scrollable asset grid.
    """

    asset_selected = pyqtSignal(object)   # AssetInfo
    asset_activated = pyqtSignal(object)  # AssetInfo
    load_selected_requested = pyqtSignal(object)  # list[AssetInfo]
    status_message = pyqtSignal(str)
    fullscreen_requested = pyqtSignal(bool)

    def __init__(self, s3_client: S3Client, parent=None):
        super().__init__(parent)
        self._client  = s3_client
        self._thumb_size = THUMB_SIZE
        self._pending_thumb_size = THUMB_SIZE
        self._assets: List[AssetInfo] = []
        self._visible_assets: List[AssetInfo] = []
        self._cards:  List[AssetCard] = []
        self._card_by_usd: Dict[str, AssetCard] = {}
        self._selected_usd_keys: Set[str] = set()
        self._last_clicked_index: Optional[int] = None
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
        self._thumbnail_size_timer = QTimer(self)
        self._thumbnail_size_timer.setSingleShot(True)
        self._thumbnail_size_timer.timeout.connect(self._apply_thumbnail_size)
        self._last_column_count = 0

        self._build_ui()
        self._connect_signals()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Header ───────────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(4)

        header = QLabel("SIMREADY ASSETS")
        header.setObjectName("section_header")
        header.setStyleSheet(
            f"color: {COLOR_TEXT_PRIMARY}; font-size: 13px; font-weight: 800; "
            f"border: none; padding: 2px 0 4px 0;"
        )
        header_row.addWidget(header, 1)

        self._fullscreen_btn = QToolButton()
        self._fullscreen_btn.setCheckable(True)
        self._fullscreen_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self._fullscreen_btn.setToolTip("Expand asset browser")
        self._fullscreen_btn.clicked.connect(self.fullscreen_requested.emit)
        self._fullscreen_btn.setStyleSheet(
            f"QToolButton {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 14px; padding: 5px; color: {COLOR_TEXT_PRIMARY}; }}"
            f"QToolButton:hover {{ background: {COLOR_BG_HOVER}; border-color: {COLOR_ACCENT_DIM}; "
            f"color: {COLOR_ACCENT}; }}"
            f"QToolButton:checked {{ background: {COLOR_ACCENT}; border-color: {COLOR_ACCENT}; "
            f"color: #090a08; }}"
        )
        header_row.addWidget(self._fullscreen_btn)
        root.addLayout(header_row)

        # ── Search bar ────────────────────────────────────────────────────────
        search_row = QHBoxLayout()
        search_row.setSpacing(4)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search assets...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._schedule_filter)
        search_row.addWidget(self._search)

        refresh_btn = QToolButton()
        refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_btn.setFixedSize(32, 32)
        refresh_btn.setToolTip("Refresh from S3")
        refresh_btn.clicked.connect(self._refresh)
        refresh_btn.setStyleSheet(
            f"QToolButton {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 16px; padding: 5px; color: {COLOR_TEXT_PRIMARY}; }}"
            f"QToolButton:hover {{ background: {COLOR_BG_HOVER}; border-color: {COLOR_ACCENT_DIM}; }}"
        )
        search_row.addWidget(refresh_btn)
        root.addLayout(search_row)

        source_add_row = QHBoxLayout()
        source_add_row.setSpacing(4)

        self._s3_location_input = QLineEdit()
        self._s3_location_input.setPlaceholderText("Add S3 location: s3://bucket/path")
        self._s3_location_input.returnPressed.connect(self._add_s3_location)
        source_add_row.addWidget(self._s3_location_input, 1)

        add_source_btn = QToolButton()
        add_source_btn.setText("+")
        add_source_btn.setFixedSize(32, 32)
        add_source_btn.setToolTip("Add S3 location")
        add_source_btn.clicked.connect(self._add_s3_location)
        add_source_btn.setStyleSheet(
            f"QToolButton {{ background: {COLOR_ACCENT}; border: none; border-radius: 16px; "
            f"color: #090a08; font-size: 15px; font-weight: 800; }}"
            f"QToolButton:hover {{ background: {COLOR_ACCENT_HOVER}; }}"
        )
        source_add_row.addWidget(add_source_btn)

        remove_source_btn = QToolButton()
        remove_source_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogDiscardButton))
        remove_source_btn.setFixedSize(32, 32)
        remove_source_btn.setToolTip("Remove the S3 location typed in the field")
        remove_source_btn.clicked.connect(self._remove_typed_s3_location)
        remove_source_btn.setStyleSheet(
            f"QToolButton {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 16px; padding: 5px; color: {COLOR_TEXT_PRIMARY}; }}"
            f"QToolButton:hover {{ background: {COLOR_BG_HOVER}; border-color: {COLOR_ACCENT_DIM}; }}"
        )
        source_add_row.addWidget(remove_source_btn)
        root.addLayout(source_add_row)

        # ── Category filter ───────────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)

        cat_label = QLabel("Folder:")
        cat_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; font-weight: 700;"
        )
        filter_row.addWidget(cat_label)

        self._cat_combo = QComboBox()
        self._cat_combo.addItem("All")
        self._cat_combo.currentTextChanged.connect(self._schedule_filter)
        filter_row.addWidget(self._cat_combo, 1)

        root.addLayout(filter_row)

        size_row = QHBoxLayout()
        size_row.setSpacing(6)

        size_label = QLabel("Thumbnail:")
        size_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px; font-weight: 700;"
        )
        size_row.addWidget(size_label)

        self._thumb_size_slider = QSlider(Qt.Horizontal)
        self._thumb_size_slider.setRange(56, 256)
        self._thumb_size_slider.setSingleStep(4)
        self._thumb_size_slider.setPageStep(16)
        self._thumb_size_slider.setTickInterval(32)
        self._thumb_size_slider.setTickPosition(QSlider.TicksBelow)
        self._thumb_size_slider.setValue(self._thumb_size)
        self._thumb_size_slider.setToolTip("Resize asset thumbnails")
        size_row.addWidget(self._thumb_size_slider, 1)

        self._thumb_size_value = QLabel(f"{self._thumb_size}px")
        self._thumb_size_value.setMinimumWidth(38)
        self._thumb_size_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._thumb_size_value.setStyleSheet(
            f"color: {COLOR_ACCENT}; font-size: 10px; font-weight: 700;"
        )
        size_row.addWidget(self._thumb_size_value)

        self._thumb_size_slider.valueChanged.connect(self._schedule_thumbnail_size)
        root.addLayout(size_row)

        # ── Asset count ───────────────────────────────────────────────────────
        self._count_label = QLabel("Loading…")
        self._count_label.setStyleSheet(
            f"color: {COLOR_TEXT_DISABLED}; font-size: 10px; padding: 1px 0;"
        )
        root.addWidget(self._count_label)

        selection_row = QHBoxLayout()
        selection_row.setSpacing(6)

        self._selection_label = QLabel("No assets selected")
        self._selection_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 10px;"
        )
        selection_row.addWidget(self._selection_label, 1)

        self._load_selected_btn = QPushButton("Load Selected")
        self._load_selected_btn.setEnabled(False)
        self._load_selected_btn.setToolTip("Load all selected assets into the viewport")
        self._load_selected_btn.clicked.connect(self._request_load_selected)
        self._load_selected_btn.setStyleSheet(
            f"QPushButton {{ background: {COLOR_ACCENT}; color: #090a08; border: none; "
            f"border-radius: 13px; padding: 6px 10px; font-size: 10px; font-weight: 800; }}"
            f"QPushButton:disabled {{ background: {COLOR_BG_PANEL}; color: {COLOR_TEXT_DISABLED}; }}"
            f"QPushButton:enabled:hover {{ background: {COLOR_ACCENT_HOVER}; color: #090a08; }}"
        )
        selection_row.addWidget(self._load_selected_btn)
        root.addLayout(selection_row)

        # ── Scrollable grid ───────────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._scroll.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: transparent;")
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(CARD_GRID_GAP)
        self._grid.setContentsMargins(2, 2, 2, 2)

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
            f"QProgressBar {{ background: {COLOR_BG_WIDGET}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: {COLOR_ACCENT}; border-radius: 3px; }}"
        )
        self._progress.hide()
        root.addWidget(self._progress)

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(BROWSER_MIN_WIDTH)
        self.setMaximumWidth(BROWSER_MAX_WIDTH)

    def set_fullscreen_mode(self, enabled: bool):
        if hasattr(self, "_fullscreen_btn"):
            self._fullscreen_btn.blockSignals(True)
            self._fullscreen_btn.setChecked(enabled)
            self._fullscreen_btn.setIcon(
                self.style().standardIcon(
                    QStyle.SP_TitleBarNormalButton if enabled else QStyle.SP_TitleBarMaxButton
                )
            )
            self._fullscreen_btn.setToolTip(
                "Restore viewport and controls" if enabled else "Expand asset browser"
            )
            self._fullscreen_btn.blockSignals(False)

        if enabled:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.setMinimumWidth(BROWSER_MIN_WIDTH)
            self.setMaximumWidth(BROWSER_MAX_WIDTH)

        self.updateGeometry()
        self._schedule_visible_thumbnail_requests()

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _schedule_thumbnail_size(self, value: int):
        self._pending_thumb_size = value
        if hasattr(self, "_thumb_size_value"):
            self._thumb_size_value.setText(f"{value}px")
        self._thumbnail_size_timer.start(120)

    def _apply_thumbnail_size(self):
        if self._pending_thumb_size == self._thumb_size:
            return

        self._thumb_size = self._pending_thumb_size
        self._last_column_count = 0
        self._render_generation += 1
        generation = self._render_generation
        self._render_index = 0
        self._render_target_index = 0
        self._requested_thumbnails.clear()
        self._thumbnail_decodes.clear()
        self._thumb_request_timer.stop()
        self._clear_grid()
        self._scroll.verticalScrollBar().setValue(0)
        self._update_selection_ui()

        if self._visible_assets:
            self._extend_render_target(self._initial_render_target())
            QTimer.singleShot(0, lambda: self._render_asset_batch(generation))

    def _connect_signals(self):
        self._client.assets_loaded.connect(self._on_assets_loaded)
        self._client.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._client.status_message.connect(self._on_status)
        self._client.error_occurred.connect(self._on_error)
        self._client.progress_updated.connect(self._on_progress)
        self._client.locations_changed.connect(self._on_locations_changed)

    def _add_s3_location(self):
        text = self._s3_location_input.text().strip()
        if not text:
            return
        try:
            added = self._client.add_location(text)
        except ValueError as exc:
            self._on_error(str(exc))
            return

        if not added:
            return
        self._s3_location_input.clear()
        self._progress.setRange(0, 0)
        self._progress.show()
        self._client.refresh(force_network=True)

    def _remove_typed_s3_location(self):
        text = self._s3_location_input.text().strip()
        if not text:
            self._on_error("Enter the S3 location you want to remove.")
            return

        try:
            source_uri = parse_s3_location(text).root_uri
        except ValueError as exc:
            self._on_error(str(exc))
            return

        if self._client.remove_location(source_uri):
            self._s3_location_input.clear()
            self._client.refresh(force_network=True)

    def _on_locations_changed(self, _locations):
        valid_sources = {location.root_uri for location in self._client.locations}
        if valid_sources:
            self._assets = [asset for asset in self._assets if asset.source_uri in valid_sources]
            valid_keys = {asset.asset_id for asset in self._assets}
            self._selected_usd_keys.intersection_update(valid_keys)
            self._update_selection_ui()
        self._update_category_filter()
        self._schedule_filter()

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _refresh(self):
        self._status.setText("Refreshing…")
        self._progress.setRange(0, 0)
        self._progress.show()
        self._render_generation += 1
        self._visible_assets = []
        self._render_index = 0
        self._selected_usd_keys.clear()
        self._last_clicked_index = None
        self._update_selection_ui()
        self._clear_grid()
        self._client.refresh(force_network=True)

    def _on_assets_loaded(self, assets: List[AssetInfo]):
        self._assets = assets
        valid_keys = {asset.asset_id for asset in assets}
        self._selected_usd_keys.intersection_update(valid_keys)
        self._update_selection_ui()
        self._update_category_filter()
        self._filter_assets()
        self._status.setText(f"{len(assets)} assets available")
        self._progress.hide()

    def _on_thumbnail_ready(self, asset: AssetInfo):
        card = self._card_by_usd.get(asset.asset_id)
        if not card or not asset.local_thumbnail:
            return

        if card.set_cached_thumbnail(asset.local_thumbnail):
            return

        thumb_size = self._thumb_size
        decode_key = (str(asset.local_thumbnail), thumb_size)
        if decode_key in self._thumbnail_decodes:
            return
        self._thumbnail_decodes.add(decode_key)
        self._thumb_decode_pool.start(
            _ThumbnailDecodeWorker(asset.asset_id, str(asset.local_thumbnail), thumb_size, self._thumb_decode_signals)
        )

    def _on_thumbnail_decoded(self, usd_key: str, path: str, size: int, image: QImage):
        self._thumbnail_decodes.discard((path, size))
        if image.isNull():
            return
        card = self._card_by_usd.get(usd_key)
        if card and size == self._thumb_size and getattr(card, "_size", size) == size:
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
        cats = sorted({a.category for a in self._assets if a.category})
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
                          or query in a.source_name.lower()
                          or query in a.source_uri.lower()
                          or any(query in t.lower() for t in a.tags))
            and (cat == "All" or a.category == cat)
        ]
        self._last_clicked_index = None

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
        self._update_selection_ui()

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
            card = AssetCard(asset, size=self._thumb_size)
            card.grid_row = row
            card.clicked.connect(self._on_card_clicked)
            card.double_clicked.connect(self._on_card_double_clicked)
            card.set_selected(asset.asset_id in self._selected_usd_keys)
            self._grid.addWidget(card, row, col)
            self._cards.append(card)
            self._card_by_usd[asset.asset_id] = card

        self._render_index = batch_end
        self._update_selection_ui()
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cols = self._column_count()
        if cols != self._last_column_count and self._visible_assets:
            self._last_column_count = cols
            self._resize_timer.start(80)
        self._schedule_visible_thumbnail_requests()

    def _column_count(self) -> int:
        margins = self._grid.contentsMargins()
        spacing = max(0, self._grid.horizontalSpacing())
        card_width = self._thumb_size + CARD_WIDTH_PAD
        available = max(1, self._scroll.viewport().width() - margins.left() - margins.right())
        return max(1, (available + spacing) // (card_width + spacing))

    def _row_step(self) -> int:
        return self._grid.verticalSpacing() + self._thumb_size + CARD_HEIGHT_PAD

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
        margin = self._thumb_size * 2
        row_step = self._row_step()

        for card in self._cards:
            card_top = self._grid.contentsMargins().top() + getattr(card, "grid_row", 0) * row_step
            card_bottom = card_top + card.height()
            if card_bottom < top - margin or card_top > bottom + margin:
                continue
            key = card.asset.asset_id + "::" + str(card.asset.thumbnail_key or card.asset.usd_key)
            if key in self._requested_thumbnails:
                continue
            self._requested_thumbnails.add(key)
            self._client.request_thumbnail(card.asset, allow_download=True)

    # ── Card selection ─────────────────────────────────────────────────────────

    def _on_card_clicked(self, asset: AssetInfo, modifiers=Qt.NoModifier):
        try:
            index = self._visible_assets.index(asset)
        except ValueError:
            index = None

        ctrl_down = bool(modifiers & Qt.ControlModifier)
        shift_down = bool(modifiers & Qt.ShiftModifier)

        if shift_down and index is not None and self._last_clicked_index is not None:
            if not ctrl_down:
                self._selected_usd_keys.clear()
            lo, hi = sorted((self._last_clicked_index, index))
            for item in self._visible_assets[lo : hi + 1]:
                self._selected_usd_keys.add(item.asset_id)
        elif ctrl_down:
            if asset.asset_id in self._selected_usd_keys:
                self._selected_usd_keys.remove(asset.asset_id)
            else:
                self._selected_usd_keys.add(asset.asset_id)
            self._last_clicked_index = index
        else:
            self._selected_usd_keys = {asset.asset_id}
            self._last_clicked_index = index

        self._update_selection_ui()
        self.asset_selected.emit(asset)

    def _on_card_double_clicked(self, asset: AssetInfo):
        self._selected_usd_keys = {asset.asset_id}
        try:
            self._last_clicked_index = self._visible_assets.index(asset)
        except ValueError:
            self._last_clicked_index = None
        self._update_selection_ui()
        self.asset_selected.emit(asset)
        self.asset_activated.emit(asset)

    def _request_load_selected(self):
        selected = self._selected_assets()
        if selected:
            self.load_selected_requested.emit(selected)

    def _selected_assets(self) -> List[AssetInfo]:
        selected = set(self._selected_usd_keys)
        return [asset for asset in self._assets if asset.asset_id in selected]

    def _update_selection_ui(self):
        selected_count = len(self._selected_usd_keys)

        if hasattr(self, "_selection_label"):
            if selected_count == 1:
                self._selection_label.setText("1 asset selected")
            elif selected_count:
                self._selection_label.setText(f"{selected_count} assets selected")
            else:
                self._selection_label.setText("No assets selected")

        if hasattr(self, "_load_selected_btn"):
            self._load_selected_btn.setEnabled(selected_count > 0)
            self._load_selected_btn.setText("Load Asset" if selected_count == 1 else "Load Selected")

        if hasattr(self, "_count_label"):
            suffix = f" | selected {selected_count}" if selected_count else ""
            self._count_label.setText(
                f"Showing {self._render_index} of {len(self._visible_assets)} filtered assets{suffix}"
            )

        for card in self._cards:
            card.set_selected(card.asset.asset_id in self._selected_usd_keys)


# ── Asset card widget ───────────────────────────────────────────────────────────

class AssetCard(QFrame):
    """Single asset thumbnail + name card in the grid."""

    clicked = pyqtSignal(object, object)  # AssetInfo, keyboard modifiers
    double_clicked = pyqtSignal(object)  # AssetInfo

    _PLACEHOLDER: Optional[QPixmap] = None
    _PIXMAP_CACHE: Dict[tuple[str, int], QPixmap] = {}
    _MAX_PIXMAP_CACHE = 256

    def __init__(self, asset: AssetInfo, size: int = THUMB_SIZE, parent=None):
        super().__init__(parent)
        self.asset = asset
        self._size = size
        self.setObjectName("asset_card")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{asset.display_name}\n{asset.source_name}\n{asset.category}\n{asset.s3_uri}")
        self.setFixedSize(size + CARD_WIDTH_PAD, size + CARD_HEIGHT_PAD)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 4)
        layout.setSpacing(4)

        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setFixedSize(size, size)
        self._thumb.setStyleSheet(
            "background: #f4f5f0; border: 1px solid #dfe6d8; border-radius: 6px;"
        )
        self._thumb.setPixmap(self._get_placeholder(size))
        layout.addWidget(self._thumb)

        name = QLabel(asset.display_name)
        name.setObjectName("asset_name")
        name.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        name.setWordWrap(False)
        name.setStyleSheet(
            f"color: {COLOR_TEXT_PRIMARY}; font-size: 9px; font-weight: 700; "
            f"background: transparent; padding: 0 2px;"
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
            self.clicked.emit(self.asset, event.modifiers())
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.asset)
            event.accept()

    def _update_style(self, selected: bool):
        if selected:
            self.setStyleSheet(
                f"QFrame#asset_card {{ background: #171f10; "
                f"border: 2px solid {COLOR_ACCENT}; border-radius: 8px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#asset_card {{ background: {COLOR_BG_PANEL}; "
                f"border: 1px solid {COLOR_BORDER}; border-radius: 8px; }}"
                f"QFrame#asset_card:hover {{ border: 1px solid {COLOR_ACCENT_DIM}; "
                f"background: {COLOR_BG_HOVER}; }}"
            )

    @classmethod
    def _get_placeholder(cls, size: int) -> QPixmap:
        if cls._PLACEHOLDER and cls._PLACEHOLDER.width() == size:
            return cls._PLACEHOLDER
        px = QPixmap(size, size)
        px.fill(QColor("#f4f5f0"))
        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)
        # Draw USD icon placeholder
        p.setPen(QPen(QColor("#c7d0c1"), 2))
        c = size // 2
        r = size // 3
        p.drawEllipse(c - r, c - r, r * 2, r * 2)
        p.setPen(QColor("#65715e"))
        p.setFont(QFont("Segoe UI", size // 8))
        p.drawText(px.rect(), Qt.AlignCenter, "USD")
        p.end()
        cls._PLACEHOLDER = px
        return px
