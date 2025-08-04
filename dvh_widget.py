# dvh_widget.py  ──────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QComboBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QColor, QIcon
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt                           # ← new
from matplotlib import colors as mcolors                  # ← new

from aligned_metrics_full import plot_all_roi_dvhs

class DVHPlotWidget(QWidget):
    """
    Composite widget that shows cumulative DVHs and lets the user
    choose which ROIs to overlay.  Each checklist entry now carries
    the same colour as its curve in the plot.
    """
    def __init__(self, dvh_abs: dict[str, "pd.DataFrame"] | None = None,
                 prescription: float | None = None,
                 parent=None):
        super().__init__(parent)
        self._dvh_abs = dvh_abs or {}
        self._prescription = prescription

        # colour bookkeeping ----------------------------------------------
        self._roi_colors: dict[str, str] = {}     # ROI name → "#rrggbb"
        self._mpl_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # ROI checklist ----------------------------------------------------
        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QListWidget.NoSelection)
        self.roi_list.itemChanged.connect(self._redraw)

        # axis selectors ---------------------------------------------------
        self.x_selector = QComboBox();  self.x_selector.addItems(["dose", "relative"])
        self.y_selector = QComboBox();  self.y_selector.addItems(["volume", "relative"])
        self.x_selector.currentIndexChanged.connect(self._redraw)
        self.y_selector.currentIndexChanged.connect(self._redraw)

        # Matplotlib canvas ------------------------------------------------
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # layout -----------------------------------------------------------
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("X‑axis:")); ctrl.addWidget(self.x_selector)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Y‑axis:")); ctrl.addWidget(self.y_selector)
        ctrl.addStretch()

        left = QVBoxLayout()
        left.addWidget(QLabel("ROIs to display:"))
        left.addWidget(self.roi_list)
        left.addLayout(ctrl)

        main = QHBoxLayout(self)
        main.addLayout(left, 0)
        main.addWidget(self.canvas, 1)

        # first fill (if data were supplied) ------------------------------
        self._populate_list()
        self._redraw()

    # =====================================================================
    def set_data(self, dvh_abs: dict[str, "pd.DataFrame"],
                 prescription: float | None):
        """Replace DVH dictionary and Rx dose, then refresh widget."""
        self._dvh_abs = dvh_abs
        self._prescription = prescription
        self._populate_list()
        self._redraw()

    # =====================================================================
    # internals
    # =====================================================================
    def _make_color_icon(self, hex_color: str) -> QIcon:
        """Return a 12×12 pixmap filled with *hex_color*."""
        pix = QPixmap(12, 12)
        pix.fill(QColor(hex_color))
        return QIcon(pix)

    def _populate_list(self):
        """Populate ROI checklist with coloured bullets (all ticked)."""
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        self._roi_colors.clear()

        for idx, roi in enumerate(sorted(self._dvh_abs.keys())):
            # deterministic colour from the Matplotlib cycle
            color = self._mpl_cycle[idx % len(self._mpl_cycle)]
            self._roi_colors[roi] = color

            item = QListWidgetItem(self._make_color_icon(color), roi)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.roi_list.addItem(item)

        self.roi_list.blockSignals(False)

    def _selected_rois(self):
        return [
            self.roi_list.item(i).text()
            for i in range(self.roi_list.count())
            if self.roi_list.item(i).checkState() == Qt.Checked
        ]

    def _redraw(self):
        self.figure.clear()
        rois = self._selected_rois()
        if not self._dvh_abs or not rois:
            self.canvas.draw_idle()
            return

        ax = self.figure.add_subplot(111)

        # set colour cycle to match the order in *rois*
        ax.set_prop_cycle('color', [self._roi_colors[r] for r in rois])

        subset = {roi: self._dvh_abs[roi] for roi in rois}
        plot_all_roi_dvhs(
            subset,
            self._prescription,
            ax=ax,
            x_mode=self.x_selector.currentText(),
            y_mode=self.y_selector.currentText(),
        )
        self.canvas.draw_idle()
