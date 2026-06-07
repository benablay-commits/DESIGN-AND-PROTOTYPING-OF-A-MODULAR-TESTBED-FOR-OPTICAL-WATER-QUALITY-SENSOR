import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt6 import QtCore, QtGui, QtWidgets


WAVELENGTHS = np.array(
    [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
    dtype=float,
)

PROFILE_RANGES = {
    "All channels": None,
    "UV profile (410-535 nm)": (0, 6),
    "White LED profile (560-705 nm)": (6, 12),
    "IR profile (730-940 nm)": (12, 18),
}

METRICS = {
    "transmission": {
        "title": "Normalized spectral response / transmission",
        "column": "Genormaliseerde respons / transmissie (-)",
        "ylabel": "Transmission (-)",
    },
    "absorbance": {
        "title": "Absorbance",
        "column": "Absorbantie A = -log10(transmissie) (-)",
        "ylabel": "Absorbance (-)",
    },
    "mean": {
        "title": "Mean spectral response",
        "column": "Gemiddelde spectrale respons (a.u.)",
        "ylabel": "Spectral response (a.u.)",
    },
    "std": {
        "title": "Spectral response standard deviation",
        "column": "Standaarddeviatie spectrale respons (a.u.)",
        "ylabel": "Standard deviation (a.u.)",
    },
    "cv": {
        "title": "Spectral response coefficient of variation",
        "column": "Variatiecoefficient spectrale respons (%)",
        "ylabel": "CV (%)",
    },
}

GRAPH_CATEGORIES = {
    "transmission": "Transmission",
    "absorbance": "Absorbance",
    "mean": "Mean spectral response",
    "std": "Standard deviation",
    "cv": "Coefficient of variation",
    "error": "Mean with SD band",
    "error_sd_subplot": "Mean+SD export: SD subplot",
    "interpolation": "Interpolated transmission",
    "linearity": "Linearity / R2",
    "spectrometer": "AS7265x vs spectrometer",
    "spectrometer_error": "Spectrometer absolute error",
}

LINEARITY_METRICS = {
    "Transmission (-)": "Genormaliseerde respons / transmissie (-)",
    "Absorbance (-)": "Absorbantie A = -log10(transmissie) (-)",
    "Mean spectral response (a.u.)": "Gemiddelde spectrale respons (a.u.)",
}


@dataclass
class SpectrumSample:
    name: str
    rows: list[dict]
    color: QtGui.QColor
    source_file: str = ""
    visible: bool = True


class CsvSpectrumAnalyzer(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Spectro CSV Analyzer")
        self.resize(1500, 950)

        self.csv_path: Path | None = None
        self.samples: list[SpectrumSample] = []
        self.loaded_files: list[Path] = []
        self.composite_parts: list[dict] = []
        self.sample_checkboxes: list[QtWidgets.QCheckBox] = []
        self.sample_color_buttons: list[QtWidgets.QPushButton] = []
        self.linearity_concentration_edits: list[QtWidgets.QLineEdit] = []
        self.linearity_last_rows: list[dict] = []
        self.metric_tables: dict[str, QtWidgets.QTableWidget] = {}
        self.spectrometer_reference_path: Path | None = None
        self.spectrometer_reference_rows: dict[int, dict] = {}
        self.spectrometer_comparison_rows: list[dict] = []
        self.active_metric = "transmission"
        self.default_colors = [
            "#111827", "#e11d48", "#2563eb", "#16a34a", "#f97316",
            "#7c3aed", "#0891b2", "#db2777", "#ca8a04", "#4b5563",
        ]
        self.line_width = 2
        self.marker_size = 7
        self.export_width = 4800
        self.title_size = "16pt"
        self.axis_label_size = "14pt"
        self.legend_text_size = "10pt"
        self.tick_font_size = 11
        self.publication_figsize = (6.8, 3.8)
        self.publication_dpi = 300
        self.graph_settings = self.default_graph_settings()

        pg.setConfigOptions(antialias=True)
        self._build_ui()

    def default_graph_settings(self) -> dict[str, dict]:
        default_colors = {
            "transmission": "#111827",
            "absorbance": "#dc2626",
            "mean": "#2563eb",
            "std": "#7c3aed",
            "cv": "#f97316",
            "error": "#2563eb",
            "error_sd_subplot": "#7c3aed",
            "interpolation": "#111827",
            "linearity": "#111827",
            "spectrometer": "#111827",
            "spectrometer_error": "#2563eb",
        }
        return {
            key: {
                "use_category_color": False,
                "color": QtGui.QColor(default_colors.get(key, "#111827")),
                "auto_x": True,
                "x_min": "",
                "x_max": "",
                "auto_y": True,
                "y_min": "",
                "y_max": "",
            }
            for key in GRAPH_CATEGORIES
        }

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #111827;
                font-family: Segoe UI;
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #111827;
                margin-top: 10px;
                padding: 8px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                background-color: white;
            }
            QPushButton, QComboBox {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #111827;
                padding: 6px 9px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
            }
            QTableWidget {
                gridline-color: #d1d5db;
                selection-background-color: #dbeafe;
            }
            QTabWidget::pane {
                border: 1px solid #111827;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #f3f4f6;
                color: #111827;
                border: 1px solid #111827;
                padding: 7px 12px;
                font-weight: 700;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #000000;
            }
            QTabBar::tab:!selected {
                color: #111827;
            }
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        side_scroll = QtWidgets.QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setMinimumWidth(360)
        side_scroll.setMaximumWidth(460)
        side_panel = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(10)
        side_scroll.setWidget(side_panel)
        root.addWidget(side_scroll)

        file_box = QtWidgets.QGroupBox("CSV files")
        file_layout = QtWidgets.QGridLayout(file_box)
        self.file_label = QtWidgets.QLabel("No CSV file loaded yet.")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label, 0, 0, 1, 2)

        open_button = QtWidgets.QPushButton("Open multiple CSV files")
        open_button.clicked.connect(self.open_csv)
        file_layout.addWidget(open_button, 1, 0)

        clear_files_button = QtWidgets.QPushButton("Clear loaded CSV files")
        clear_files_button.clicked.connect(self.clear_loaded_data)
        file_layout.addWidget(clear_files_button, 1, 1)

        self.export_png_button = QtWidgets.QPushButton("Export current graph as PNG")
        self.export_png_button.clicked.connect(self.export_current_png)
        file_layout.addWidget(self.export_png_button, 2, 0)

        export_csv_button = QtWidgets.QPushButton("Export processed CSV")
        export_csv_button.clicked.connect(self.export_filtered_csv)
        file_layout.addWidget(export_csv_button, 2, 1)

        export_table_button = QtWidgets.QPushButton("Export current table CSV")
        export_table_button.clicked.connect(self.export_current_table_csv)
        file_layout.addWidget(export_table_button, 3, 0, 1, 2)
        side_layout.addWidget(file_box)

        plot_box = QtWidgets.QGroupBox("Graph options")
        plot_layout = QtWidgets.QGridLayout(plot_box)
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.addItems(PROFILE_RANGES.keys())
        self.profile_combo.currentIndexChanged.connect(self.refresh_all_plots)
        plot_layout.addWidget(QtWidgets.QLabel("Profile filter"), 0, 0)
        plot_layout.addWidget(self.profile_combo, 0, 1)

        self.show_points_checkbox = QtWidgets.QCheckBox("Show measurement points")
        self.show_points_checkbox.setChecked(True)
        self.show_points_checkbox.toggled.connect(self.refresh_all_plots)
        plot_layout.addWidget(self.show_points_checkbox, 1, 0, 1, 2)

        self.show_grid_checkbox = QtWidgets.QCheckBox("Show grid")
        self.show_grid_checkbox.setChecked(True)
        self.show_grid_checkbox.toggled.connect(self.refresh_all_plots)
        plot_layout.addWidget(self.show_grid_checkbox, 2, 0, 1, 2)
        side_layout.addWidget(plot_box)

        style_box = QtWidgets.QGroupBox("Graph category style and scale")
        style_layout = QtWidgets.QGridLayout(style_box)
        self.graph_category_combo = QtWidgets.QComboBox()
        for key, label in GRAPH_CATEGORIES.items():
            self.graph_category_combo.addItem(label, key)
        self.graph_category_combo.currentIndexChanged.connect(self.load_graph_setting_controls)
        style_layout.addWidget(QtWidgets.QLabel("Graph category"), 0, 0)
        style_layout.addWidget(self.graph_category_combo, 0, 1, 1, 3)

        self.use_category_color_checkbox = QtWidgets.QCheckBox("Use one fixed color for this category")
        self.use_category_color_checkbox.toggled.connect(self.save_graph_setting_controls)
        style_layout.addWidget(self.use_category_color_checkbox, 1, 0, 1, 4)

        self.category_color_button = QtWidgets.QPushButton("Choose category color")
        self.category_color_button.clicked.connect(self.choose_graph_category_color)
        style_layout.addWidget(self.category_color_button, 2, 0, 1, 4)

        self.auto_x_checkbox = QtWidgets.QCheckBox("Automatic X scale")
        self.auto_x_checkbox.toggled.connect(self.save_graph_setting_controls)
        style_layout.addWidget(self.auto_x_checkbox, 3, 0, 1, 4)
        self.x_min_edit = QtWidgets.QLineEdit()
        self.x_max_edit = QtWidgets.QLineEdit()
        self.x_min_edit.setPlaceholderText("X min")
        self.x_max_edit.setPlaceholderText("X max")
        style_layout.addWidget(QtWidgets.QLabel("X range"), 4, 0)
        style_layout.addWidget(self.x_min_edit, 4, 1)
        style_layout.addWidget(QtWidgets.QLabel("to"), 4, 2)
        style_layout.addWidget(self.x_max_edit, 4, 3)

        self.auto_y_checkbox = QtWidgets.QCheckBox("Automatic Y scale")
        self.auto_y_checkbox.toggled.connect(self.save_graph_setting_controls)
        style_layout.addWidget(self.auto_y_checkbox, 5, 0, 1, 4)
        self.y_min_edit = QtWidgets.QLineEdit()
        self.y_max_edit = QtWidgets.QLineEdit()
        self.y_min_edit.setPlaceholderText("Y min")
        self.y_max_edit.setPlaceholderText("Y max")
        style_layout.addWidget(QtWidgets.QLabel("Y range"), 6, 0)
        style_layout.addWidget(self.y_min_edit, 6, 1)
        style_layout.addWidget(QtWidgets.QLabel("to"), 6, 2)
        style_layout.addWidget(self.y_max_edit, 6, 3)

        apply_style_button = QtWidgets.QPushButton("Apply style and scale")
        apply_style_button.clicked.connect(self.apply_graph_style_controls)
        style_layout.addWidget(apply_style_button, 7, 0, 1, 2)
        reset_style_button = QtWidgets.QPushButton("Reset selected category")
        reset_style_button.clicked.connect(self.reset_current_graph_setting)
        style_layout.addWidget(reset_style_button, 7, 2, 1, 2)
        side_layout.addWidget(style_box)
        self.load_graph_setting_controls()

        builder_box = QtWidgets.QGroupBox("Build composite line")
        builder_layout = QtWidgets.QGridLayout(builder_box)

        self.builder_source_combo = QtWidgets.QComboBox()
        self.builder_source_combo.currentIndexChanged.connect(self.refresh_builder_sample_combo)
        builder_layout.addWidget(QtWidgets.QLabel("CSV source"), 0, 0)
        builder_layout.addWidget(self.builder_source_combo, 0, 1)

        self.builder_sample_combo = QtWidgets.QComboBox()
        builder_layout.addWidget(QtWidgets.QLabel("Sample"), 1, 0)
        builder_layout.addWidget(self.builder_sample_combo, 1, 1)

        self.builder_profile_combo = QtWidgets.QComboBox()
        self.builder_profile_combo.addItems([
            "UV profile (410-535 nm)",
            "White LED profile (560-705 nm)",
            "IR profile (730-940 nm)",
            "All channels",
        ])
        builder_layout.addWidget(QtWidgets.QLabel("Profile segment"), 2, 0)
        builder_layout.addWidget(self.builder_profile_combo, 2, 1)

        add_part_button = QtWidgets.QPushButton("Add segment")
        add_part_button.clicked.connect(self.add_composite_part)
        builder_layout.addWidget(add_part_button, 3, 0, 1, 2)

        self.composite_parts_list = QtWidgets.QListWidget()
        self.composite_parts_list.setMinimumHeight(90)
        builder_layout.addWidget(self.composite_parts_list, 4, 0, 1, 2)

        remove_part_button = QtWidgets.QPushButton("Remove selected segment")
        remove_part_button.clicked.connect(self.remove_selected_composite_part)
        builder_layout.addWidget(remove_part_button, 5, 0, 1, 2)

        self.composite_name_edit = QtWidgets.QLineEdit("Composite sample")
        builder_layout.addWidget(QtWidgets.QLabel("New line name"), 6, 0)
        builder_layout.addWidget(self.composite_name_edit, 6, 1)

        self.composite_color_button = QtWidgets.QPushButton("Composite line color")
        self.composite_color = QtGui.QColor("#111827")
        self.composite_color_button.setStyleSheet("background-color: #111827; color: white; border: 1px solid #111827;")
        self.composite_color_button.clicked.connect(self.choose_composite_color)
        builder_layout.addWidget(self.composite_color_button, 7, 0, 1, 2)

        create_composite_button = QtWidgets.QPushButton("Create composite line")
        create_composite_button.clicked.connect(self.create_composite_sample)
        builder_layout.addWidget(create_composite_button, 8, 0, 1, 2)

        clear_parts_button = QtWidgets.QPushButton("Clear segment list")
        clear_parts_button.clicked.connect(self.clear_composite_parts)
        builder_layout.addWidget(clear_parts_button, 9, 0, 1, 2)

        side_layout.addWidget(builder_box)

        sample_box = QtWidgets.QGroupBox("Samples")
        sample_layout = QtWidgets.QVBoxLayout(sample_box)
        self.sample_list_widget = QtWidgets.QWidget()
        self.sample_list_layout = QtWidgets.QVBoxLayout(self.sample_list_widget)
        self.sample_list_layout.setContentsMargins(0, 0, 0, 0)
        self.sample_list_layout.setSpacing(6)
        sample_layout.addWidget(self.sample_list_widget)

        sample_buttons = QtWidgets.QGridLayout()
        select_all = QtWidgets.QPushButton("Show all")
        select_all.clicked.connect(lambda: self.set_all_samples_visible(True))
        sample_buttons.addWidget(select_all, 0, 0)
        hide_all = QtWidgets.QPushButton("Hide all")
        hide_all.clicked.connect(lambda: self.set_all_samples_visible(False))
        sample_buttons.addWidget(hide_all, 0, 1)
        sample_layout.addLayout(sample_buttons)

        remove_layout = QtWidgets.QGridLayout()
        self.sample_remove_combo = QtWidgets.QComboBox()
        self.sample_remove_combo.addItem("No lines loaded", -1)
        remove_layout.addWidget(QtWidgets.QLabel("Remove line"), 0, 0)
        remove_layout.addWidget(self.sample_remove_combo, 0, 1)
        remove_button = QtWidgets.QPushButton("Remove selected line")
        remove_button.clicked.connect(self.remove_selected_sample)
        remove_layout.addWidget(remove_button, 1, 0, 1, 2)
        sample_layout.addLayout(remove_layout)
        side_layout.addWidget(sample_box, stretch=1)

        info_box = QtWidgets.QGroupBox("Interpretation")
        info_layout = QtWidgets.QVBoxLayout(info_box)
        info = QtWidgets.QLabel(
            "Use transmission to evaluate relative light passage, absorbance to evaluate absorption behaviour, "
            "and SD/CV to assess channel stability and repeatability."
        )
        info.setWordWrap(True)
        info_layout.addWidget(info)
        side_layout.addWidget(info_box)
        side_layout.addStretch(1)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.plot_widgets: dict[str, pg.PlotWidget] = {}
        self.error_plot = None
        self.error_table = None
        self.interpolation_plot = None
        self.interpolation_table = None
        self.linearity_plot = None
        self.linearity_table = None
        self.spectrometer_compare_plot = None
        self.spectrometer_error_plot = None

        for metric_key, meta in METRICS.items():
            plot = self._create_plot(meta["title"], meta["ylabel"])
            table = self._create_data_table()
            self.plot_widgets[metric_key] = plot
            self.metric_tables[metric_key] = table
            self.tabs.addTab(self._build_plot_table_tab(plot, table), meta["title"])

        self.error_plot = self._create_plot("Mean spectral response", "Spectral response (a.u.)")
        self.error_table = self._create_data_table()
        self.tabs.addTab(self._build_error_tab(), "Mean ± SD")

        self.interpolation_plot = self._create_plot("Interpolated transmission", "Transmission (-)")
        self.interpolation_table = self._create_data_table()
        self.tabs.addTab(self._build_plot_table_tab(self.interpolation_plot, self.interpolation_table), "Interpolation")

        self.linearity_tab = self._build_linearity_tab()
        self.tabs.addTab(self.linearity_tab, "Linearity / R2")

        self.spectrometer_tab = self._build_spectrometer_comparison_tab()
        self.tabs.addTab(self.spectrometer_tab, "Compare with spectrometer")

        self.table = QtWidgets.QTableWidget()
        self.tabs.addTab(self.table, "CSV table")

    def _build_linearity_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Use this tab for an indicative linearity test with three concentration points plus a blank/reference as the zero point. "
            "Enter the concentration for each sample, choose a parameter and wavelength, then calculate R2."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QtWidgets.QGroupBox("Linearity settings")
        controls_layout = QtWidgets.QGridLayout(controls)

        self.linearity_metric_combo = QtWidgets.QComboBox()
        self.linearity_metric_combo.addItems(LINEARITY_METRICS.keys())
        controls_layout.addWidget(QtWidgets.QLabel("Parameter"), 0, 0)
        controls_layout.addWidget(self.linearity_metric_combo, 0, 1)

        self.linearity_wavelength_combo = QtWidgets.QComboBox()
        for wavelength in WAVELENGTHS:
            self.linearity_wavelength_combo.addItem(f"{int(wavelength)} nm", int(wavelength))
        controls_layout.addWidget(QtWidgets.QLabel("Wavelength"), 0, 2)
        controls_layout.addWidget(self.linearity_wavelength_combo, 0, 3)

        self.linearity_unit_edit = QtWidgets.QLineEdit("g/L")
        controls_layout.addWidget(QtWidgets.QLabel("Concentration unit"), 1, 0)
        controls_layout.addWidget(self.linearity_unit_edit, 1, 1)

        calculate_button = QtWidgets.QPushButton("Calculate linearity / R2")
        calculate_button.clicked.connect(self.calculate_linearity)
        controls_layout.addWidget(calculate_button, 1, 2, 1, 2)

        blank_button = QtWidgets.QPushButton("Select blank/reference document and set to 0")
        blank_button.clicked.connect(self.set_blank_references_to_zero)
        controls_layout.addWidget(blank_button, 2, 0, 1, 2)

        export_png_button = QtWidgets.QPushButton("Export linearity PNG")
        export_png_button.clicked.connect(self.export_linearity_png)
        controls_layout.addWidget(export_png_button, 2, 2)

        export_csv_button = QtWidgets.QPushButton("Export linearity CSV")
        export_csv_button.clicked.connect(self.export_linearity_csv)
        controls_layout.addWidget(export_csv_button, 2, 3)

        layout.addWidget(controls)

        self.linearity_samples_box = QtWidgets.QGroupBox("Concentrations per sample")
        self.linearity_samples_layout = QtWidgets.QVBoxLayout(self.linearity_samples_box)
        self.linearity_samples_layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.linearity_samples_box)

        self.linearity_result_label = QtWidgets.QLabel(
            "No calculation yet. Tip: use the blank/reference as the zero point and preferably use absorbance at the most responsive channel."
        )
        self.linearity_result_label.setWordWrap(True)
        layout.addWidget(self.linearity_result_label)

        self.linearity_plot = self._create_plot("Linearity: response versus concentration", "Response")
        self.set_axis_label(self.linearity_plot, "bottom", "Concentration")
        self.linearity_table = self._create_data_table()
        layout.addWidget(self._build_plot_table_tab(self.linearity_plot, self.linearity_table), stretch=1)

        return tab

    def _build_spectrometer_comparison_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Compare an AS7265x CSV with a blank-corrected reference spectrometer CSV. "
            "Only overlapping AS7265x channels are used; 940 nm is ignored automatically when the reference data end at 900 nm."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QtWidgets.QGroupBox("Reference spectrometer comparison")
        controls_layout = QtWidgets.QGridLayout(controls)

        self.spectrometer_file_label = QtWidgets.QLabel("No reference spectrometer CSV loaded yet.")
        self.spectrometer_file_label.setWordWrap(True)
        controls_layout.addWidget(self.spectrometer_file_label, 0, 0, 1, 4)

        open_reference_button = QtWidgets.QPushButton("Open corrected spectrometer CSV")
        open_reference_button.clicked.connect(self.open_spectrometer_reference_csv)
        controls_layout.addWidget(open_reference_button, 1, 0, 1, 2)

        self.spectrometer_sample_combo = QtWidgets.QComboBox()
        controls_layout.addWidget(QtWidgets.QLabel("AS7265x sample"), 2, 0)
        controls_layout.addWidget(self.spectrometer_sample_combo, 2, 1, 1, 3)

        self.spectrometer_metric_combo = QtWidgets.QComboBox()
        self.spectrometer_metric_combo.addItems(["Transmission (-)", "Absorbance (-)"])
        controls_layout.addWidget(QtWidgets.QLabel("Parameter"), 3, 0)
        controls_layout.addWidget(self.spectrometer_metric_combo, 3, 1)

        compare_button = QtWidgets.QPushButton("Compare with spectrometer")
        compare_button.clicked.connect(self.compare_with_spectrometer)
        controls_layout.addWidget(compare_button, 3, 2, 1, 2)

        export_png_button = QtWidgets.QPushButton("Export comparison PNG")
        export_png_button.clicked.connect(self.export_spectrometer_comparison_png)
        controls_layout.addWidget(export_png_button, 4, 0, 1, 2)

        export_csv_button = QtWidgets.QPushButton("Export error table CSV")
        export_csv_button.clicked.connect(self.export_spectrometer_comparison_csv)
        controls_layout.addWidget(export_csv_button, 4, 2, 1, 2)

        layout.addWidget(controls)

        self.spectrometer_result_label = QtWidgets.QLabel(
            "No comparison yet. First load your AS7265x CSV using the main button on the left, then load a corrected spectrometer CSV here."
        )
        self.spectrometer_result_label.setWordWrap(True)
        layout.addWidget(self.spectrometer_result_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.spectrometer_compare_plot = self._create_plot("AS7265x versus reference spectrometer", "Value")
        self.spectrometer_error_plot = self._create_plot("Error per AS7265x channel", "Absolute error")
        splitter.addWidget(self.spectrometer_compare_plot)
        splitter.addWidget(self.spectrometer_error_plot)
        splitter.setSizes([520, 300])
        layout.addWidget(splitter, stretch=1)

        self.spectrometer_table = QtWidgets.QTableWidget()
        self.spectrometer_table.setMaximumHeight(220)
        layout.addWidget(self.spectrometer_table)

        return tab

    def _create_plot(self, title: str, y_label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title)
        plot.setMinimumHeight(560)
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.08)
        plot.setTitle(title, size=self.title_size, color="#111827")
        self.set_axis_label(plot, "bottom", "Wavelength (nm)")
        self.set_axis_label(plot, "left", y_label)
        plot.getAxis("bottom").setPen(pg.mkPen("#111827", width=3))
        plot.getAxis("left").setPen(pg.mkPen("#111827", width=3))
        plot.getAxis("bottom").setTextPen(pg.mkPen("#111827"))
        plot.getAxis("left").setTextPen(pg.mkPen("#111827"))
        tick_font = QtGui.QFont("Arial", self.tick_font_size)
        tick_font.setBold(True)
        plot.getAxis("bottom").setStyle(tickFont=tick_font, tickTextOffset=10)
        plot.getAxis("left").setStyle(tickFont=tick_font, tickTextOffset=10)
        plot.getViewBox().setDefaultPadding(0.10)
        return plot

    def set_axis_label(self, plot: pg.PlotWidget, axis_name: str, text: str) -> None:
        axis = plot.getAxis(axis_name)
        axis.setLabel(
            text,
            color="#111827",
            **{
                "font-size": self.axis_label_size,
                "font-weight": "800",
                "font-family": "Arial",
            },
        )
        label_font = QtGui.QFont("Arial", int(self.axis_label_size.removesuffix("pt")))
        label_font.setBold(True)
        axis.label.setFont(label_font)
        axis.label.setDefaultTextColor(QtGui.QColor("#111827"))

    def _create_data_table(self) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget()
        table.setMinimumHeight(180)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        return table

    def _build_error_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        btn_row = QtWidgets.QHBoxLayout()
        export_sd_btn = QtWidgets.QPushButton("Export mean + SD subplot PNG (2 panels)")
        export_sd_btn.clicked.connect(self.export_error_subplot_png)
        btn_row.addWidget(export_sd_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.error_plot)
        splitter.addWidget(self.error_table)
        splitter.setSizes([700, 220])
        layout.addWidget(splitter)
        return tab

    def _build_plot_table_tab(self, plot: pg.PlotWidget, table: QtWidgets.QTableWidget) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(plot)
        splitter.addWidget(table)
        splitter.setSizes([700, 220])
        layout.addWidget(splitter)
        return tab

    def add_readable_legend(self, plot: pg.PlotWidget, offset: tuple[int, int] = (-18, 18)) -> None:
        legend = plot.addLegend(
            offset=offset,
            labelTextColor="#111827",
            labelTextSize=self.legend_text_size,
        )
        legend.anchor((1, 0), (1, 0), offset=offset)
        legend.setBrush(pg.mkBrush(255, 255, 255, 235))
        legend.setPen(pg.mkPen("#111827", width=2))
        legend.setZValue(1000)
        try:
            legend.layout.setSpacing(12)
        except AttributeError:
            pass

    def legend_label(self, sample: SpectrumSample) -> str:
        # Keep legends thesis-readable by removing the source CSV prefix.
        label = sample.name.split("|", 1)[-1].strip()
        return re.sub(r"\.csv\b", "", label, flags=re.IGNORECASE)

    def current_graph_category_key(self) -> str:
        if not hasattr(self, "graph_category_combo"):
            return "transmission"
        return self.graph_category_combo.currentData() or "transmission"

    def graph_setting(self, category_key: str) -> dict:
        return self.graph_settings.setdefault(category_key, self.default_graph_settings().get(category_key, {}))

    def load_graph_setting_controls(self) -> None:
        if not hasattr(self, "graph_category_combo"):
            return
        setting = self.graph_setting(self.current_graph_category_key())
        self.use_category_color_checkbox.blockSignals(True)
        self.auto_x_checkbox.blockSignals(True)
        self.auto_y_checkbox.blockSignals(True)
        self.use_category_color_checkbox.setChecked(bool(setting.get("use_category_color", False)))
        self.auto_x_checkbox.setChecked(bool(setting.get("auto_x", True)))
        self.auto_y_checkbox.setChecked(bool(setting.get("auto_y", True)))
        self.use_category_color_checkbox.blockSignals(False)
        self.auto_x_checkbox.blockSignals(False)
        self.auto_y_checkbox.blockSignals(False)
        self.x_min_edit.setText(str(setting.get("x_min", "")))
        self.x_max_edit.setText(str(setting.get("x_max", "")))
        self.y_min_edit.setText(str(setting.get("y_min", "")))
        self.y_max_edit.setText(str(setting.get("y_max", "")))
        self.update_category_color_button()

    def save_graph_setting_controls(self) -> None:
        if not hasattr(self, "graph_category_combo"):
            return
        setting = self.graph_setting(self.current_graph_category_key())
        setting["use_category_color"] = self.use_category_color_checkbox.isChecked()
        setting["auto_x"] = self.auto_x_checkbox.isChecked()
        setting["x_min"] = self.x_min_edit.text().strip()
        setting["x_max"] = self.x_max_edit.text().strip()
        setting["auto_y"] = self.auto_y_checkbox.isChecked()
        setting["y_min"] = self.y_min_edit.text().strip()
        setting["y_max"] = self.y_max_edit.text().strip()
        self.set_scale_edits_enabled()

    def set_scale_edits_enabled(self) -> None:
        x_enabled = not self.auto_x_checkbox.isChecked()
        y_enabled = not self.auto_y_checkbox.isChecked()
        self.x_min_edit.setEnabled(x_enabled)
        self.x_max_edit.setEnabled(x_enabled)
        self.y_min_edit.setEnabled(y_enabled)
        self.y_max_edit.setEnabled(y_enabled)

    def update_category_color_button(self) -> None:
        setting = self.graph_setting(self.current_graph_category_key())
        color = setting.get("color", QtGui.QColor("#111827"))
        self.category_color_button.setStyleSheet(
            f"background-color: {color.name()}; color: white; border: 1px solid #111827; font-weight: 700;"
        )
        self.set_scale_edits_enabled()

    def choose_graph_category_color(self) -> None:
        setting = self.graph_setting(self.current_graph_category_key())
        color = QtWidgets.QColorDialog.getColor(setting.get("color", QtGui.QColor("#111827")), self, "Choose graph category color")
        if not color.isValid():
            return
        setting["color"] = color
        setting["use_category_color"] = True
        self.use_category_color_checkbox.setChecked(True)
        self.update_category_color_button()
        self.refresh_all_plots()

    def apply_graph_style_controls(self) -> None:
        self.save_graph_setting_controls()
        self.refresh_all_plots()

    def reset_current_graph_setting(self) -> None:
        category_key = self.current_graph_category_key()
        self.graph_settings[category_key] = self.default_graph_settings()[category_key]
        self.load_graph_setting_controls()
        self.refresh_all_plots()

    def category_color_for_sample(self, sample: SpectrumSample, category_key: str) -> QtGui.QColor:
        setting = self.graph_setting(category_key)
        if setting.get("use_category_color", False):
            return QtGui.QColor(setting.get("color", QtGui.QColor("#111827")))
        return QtGui.QColor(sample.color)

    def parsed_range_value(self, text: str) -> float | None:
        value = self.to_float(text.replace(",", "."))
        return float(value) if np.isfinite(value) else None

    def manual_axis_range(self, category_key: str, axis: str) -> tuple[float, float] | None:
        setting = self.graph_setting(category_key)
        if setting.get(f"auto_{axis}", True):
            return None
        lower = self.parsed_range_value(str(setting.get(f"{axis}_min", "")))
        upper = self.parsed_range_value(str(setting.get(f"{axis}_max", "")))
        if lower is None or upper is None or upper <= lower:
            return None
        return lower, upper

    def apply_graph_scale(self, plot: pg.PlotWidget, values: list[float], category_key: str, metric_key: str | None = None) -> None:
        x_range = self.manual_axis_range(category_key, "x")
        y_range = self.manual_axis_range(category_key, "y")
        if x_range is not None:
            plot.setXRange(x_range[0], x_range[1], padding=0)
        elif category_key not in {"linearity", "spectrometer", "spectrometer_error"}:
            indices = self.profile_indices()
            if indices.size:
                plot.setXRange(float(WAVELENGTHS[indices[0]]) - 10, float(WAVELENGTHS[indices[-1]]) + 10, padding=0)
        if y_range is not None:
            plot.setYRange(y_range[0], y_range[1], padding=0)
        else:
            self.apply_thesis_y_range(plot, values, metric_key)

    def apply_publication_scale(self, ax, values: list[float], category_key: str, metric_key: str | None = None) -> None:
        x_range = self.manual_axis_range(category_key, "x")
        y_range = self.manual_axis_range(category_key, "y")
        if x_range is not None:
            ax.set_xlim(*x_range)
        elif category_key not in {"linearity", "spectrometer", "spectrometer_error"}:
            indices = self.profile_indices()
            if indices.size:
                ax.set_xlim(float(WAVELENGTHS[indices[0]]) - 12, float(WAVELENGTHS[indices[-1]]) + 12)
        if y_range is not None:
            ax.set_ylim(*y_range)
        else:
            self.apply_publication_y_range(ax, values, metric_key)

    def open_csv(self) -> None:
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open one or more spectro CSV files",
            str(Path(__file__).resolve().parent),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not file_names:
            return

        new_samples = []
        loaded_paths = []
        try:
            for file_name in file_names:
                path = Path(file_name)
                new_samples.extend(self.load_samples_from_csv(path))
                loaded_paths.append(path)
        except (OSError, csv.Error, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "CSV loading failed", str(exc))
            return

        start_index = len(self.samples)
        for offset, sample in enumerate(new_samples):
            sample.color = QtGui.QColor(self.default_colors[(start_index + offset) % len(self.default_colors)])
        self.samples.extend(new_samples)
        self.loaded_files.extend(loaded_paths)
        self.csv_path = loaded_paths[-1] if loaded_paths else None
        self.file_label.setText(f"{len(self.loaded_files)} CSV file(s) loaded.")
        self.rebuild_sample_controls()
        self.refresh_builder_source_combo()
        self.populate_table()
        self.refresh_all_plots()

    def load_samples_from_csv(self, path: Path) -> list[SpectrumSample]:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            if not reader.fieldnames:
                raise ValueError("CSV has no header.")
            rows = list(reader)

        if not rows:
            raise ValueError("CSV contains no data.")
        if "Sample naam" not in rows[0] or "Golflengte (nm)" not in rows[0]:
            raise ValueError("CSV is missing at least 'Sample naam' or 'Golflengte (nm)'.")

        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row.get("Sample naam", "Unknown") or "Unknown", []).append(row)

        samples = []
        for index, (name, sample_rows) in enumerate(grouped.items()):
            color = QtGui.QColor(self.default_colors[index % len(self.default_colors)])
            display_name = f"{path.name} | {name}"
            samples.append(SpectrumSample(name=display_name, rows=sample_rows, color=color, source_file=path.name))
        return samples

    def clear_loaded_data(self) -> None:
        self.samples.clear()
        self.loaded_files.clear()
        self.composite_parts.clear()
        self.file_label.setText("No CSV file loaded yet.")
        self.rebuild_sample_controls()
        self.refresh_builder_source_combo()
        self.refresh_composite_parts_list()
        self.populate_table()
        self.refresh_all_plots()

    def rebuild_sample_controls(self) -> None:
        while self.sample_list_layout.count():
            item = self.sample_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.sample_checkboxes.clear()
        self.sample_color_buttons.clear()

        for sample in self.samples:
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            checkbox = QtWidgets.QCheckBox(sample.name)
            checkbox.setChecked(sample.visible)
            checkbox.toggled.connect(lambda checked, s=sample: self.set_sample_visible(s, checked))
            layout.addWidget(checkbox, stretch=1)

            color_button = QtWidgets.QPushButton("Color")
            color_button.setStyleSheet(f"background-color: {sample.color.name()}; color: white; border: 1px solid #111827;")
            color_button.clicked.connect(lambda _checked=False, s=sample, b=color_button: self.choose_sample_color(s, b))
            layout.addWidget(color_button)

            self.sample_list_layout.addWidget(row)
            self.sample_checkboxes.append(checkbox)
            self.sample_color_buttons.append(color_button)
        self.sample_list_layout.addStretch(1)
        self.refresh_sample_remove_combo()
        self.refresh_linearity_sample_controls()
        self.refresh_spectrometer_sample_combo()

    def refresh_sample_remove_combo(self) -> None:
        if not hasattr(self, "sample_remove_combo"):
            return
        current_index = self.sample_remove_combo.currentData()
        self.sample_remove_combo.blockSignals(True)
        self.sample_remove_combo.clear()
        if not self.samples:
            self.sample_remove_combo.addItem("No lines loaded", -1)
        else:
            for index, sample in enumerate(self.samples):
                self.sample_remove_combo.addItem(sample.name, index)
            if isinstance(current_index, int) and 0 <= current_index < self.sample_remove_combo.count():
                self.sample_remove_combo.setCurrentIndex(current_index)
        self.sample_remove_combo.blockSignals(False)

    def refresh_spectrometer_sample_combo(self) -> None:
        if not hasattr(self, "spectrometer_sample_combo"):
            return
        current = self.spectrometer_sample_combo.currentData()
        self.spectrometer_sample_combo.blockSignals(True)
        self.spectrometer_sample_combo.clear()
        if not self.samples:
            self.spectrometer_sample_combo.addItem("No AS7265x samples loaded", -1)
        else:
            for index, sample in enumerate(self.samples):
                self.spectrometer_sample_combo.addItem(sample.name, index)
            if isinstance(current, int) and 0 <= current < len(self.samples):
                self.spectrometer_sample_combo.setCurrentIndex(current)
        self.spectrometer_sample_combo.blockSignals(False)

    def refresh_linearity_sample_controls(self) -> None:
        if not hasattr(self, "linearity_samples_layout"):
            return
        previous_values = {}
        for edit in self.linearity_concentration_edits:
            sample_name = edit.property("sample_name")
            if sample_name:
                previous_values[str(sample_name)] = edit.text()

        while self.linearity_samples_layout.count():
            item = self.linearity_samples_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.linearity_concentration_edits.clear()

        if not self.samples:
            self.linearity_samples_layout.addWidget(QtWidgets.QLabel("Load CSV data before entering concentrations."))
            return

        header = QtWidgets.QLabel("Enter values only for the samples that should be included in the regression.")
        header.setWordWrap(True)
        self.linearity_samples_layout.addWidget(header)

        for sample in self.samples:
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            label = QtWidgets.QLabel(sample.name)
            label.setMinimumWidth(220)
            layout.addWidget(label, stretch=1)

            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("bv. 0, 1, 2.5, 5")
            edit.setProperty("sample_name", sample.name)
            edit.setText(previous_values.get(sample.name, self.guess_concentration_from_name(sample.name)))
            layout.addWidget(edit)
            self.linearity_samples_layout.addWidget(row)
            self.linearity_concentration_edits.append(edit)

        self.linearity_samples_layout.addStretch(1)

    def guess_concentration_from_name(self, name: str) -> str:
        if self.is_blank_reference_name(name):
            return "0"
        match = re.search(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:g\s*/\s*l|g/l|mg\s*/\s*l|mg/l)", name, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", ".")
        return ""

    def is_blank_reference_name(self, name: str) -> bool:
        lowered = name.lower()
        blank_keywords = [
            "blank",
            "blanko",
            "blanco",
            "reference",
            "referentie",
            "ref ",
            " ref",
            "distilled water",
            "gedestilleerd",
            "gedistilleerd",
            "demi water",
            "demineralized",
        ]
        return any(keyword in lowered for keyword in blank_keywords)

    def set_blank_references_to_zero(self) -> None:
        if not self.samples:
            QtWidgets.QMessageBox.warning(self, "Blank/reference", "Load the CSV documents first.")
            return

        sample_names = [sample.name for sample in self.samples]
        suggested_index = 0
        for index, sample in enumerate(self.samples):
            if self.is_blank_reference_name(sample.name):
                suggested_index = index
                break

        selected_name, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Choose blank/reference document",
            "Select the CSV document/sample that should be used as the blank/reference:",
            sample_names,
            suggested_index,
            False,
        )
        if not accepted or not selected_name:
            return

        for sample, edit in zip(self.samples, self.linearity_concentration_edits):
            if sample.name == selected_name:
                edit.setText("0")
                self.linearity_result_label.setText(
                    f"Blank/reference set as zero-concentration point: {selected_name}"
                )
                return

    def set_sample_visible(self, sample: SpectrumSample, visible: bool) -> None:
        sample.visible = visible
        self.refresh_all_plots()

    def set_all_samples_visible(self, visible: bool) -> None:
        for sample, checkbox in zip(self.samples, self.sample_checkboxes):
            sample.visible = visible
            checkbox.blockSignals(True)
            checkbox.setChecked(visible)
            checkbox.blockSignals(False)
        self.refresh_all_plots()

    def remove_selected_sample(self) -> None:
        selected_index = self.sample_remove_combo.currentData()
        if selected_index is None or selected_index < 0 or selected_index >= len(self.samples):
            QtWidgets.QMessageBox.warning(self, "No line selected", "Select a line before removing it.")
            return

        removed_name = self.samples[selected_index].name
        self.samples.pop(selected_index)
        updated_parts = []
        for part in self.composite_parts:
            part_index = part["sample_index"]
            if part_index == selected_index:
                continue
            if part_index > selected_index:
                part = dict(part)
                part["sample_index"] = part_index - 1
            updated_parts.append(part)
        self.composite_parts = updated_parts

        self.rebuild_sample_controls()
        self.refresh_builder_source_combo()
        self.refresh_composite_parts_list()
        self.populate_table()
        self.refresh_all_plots()
        self.file_label.setText(f"Removed line: {removed_name}")

    def choose_sample_color(self, sample: SpectrumSample, button: QtWidgets.QPushButton) -> None:
        color = QtWidgets.QColorDialog.getColor(sample.color, self, f"Choose color for {sample.name}")
        if not color.isValid():
            return
        sample.color = color
        button.setStyleSheet(f"background-color: {sample.color.name()}; color: white; border: 1px solid #111827;")
        self.refresh_all_plots()

    def refresh_builder_source_combo(self) -> None:
        if not hasattr(self, "builder_source_combo"):
            return
        current = self.builder_source_combo.currentText()
        source_names = sorted({sample.source_file for sample in self.samples if sample.source_file})
        self.builder_source_combo.blockSignals(True)
        self.builder_source_combo.clear()
        self.builder_source_combo.addItems(source_names)
        if current in source_names:
            self.builder_source_combo.setCurrentText(current)
        self.builder_source_combo.blockSignals(False)
        self.refresh_builder_sample_combo()

    def refresh_builder_sample_combo(self) -> None:
        if not hasattr(self, "builder_sample_combo"):
            return
        source_name = self.builder_source_combo.currentText()
        self.builder_sample_combo.clear()
        for index, sample in enumerate(self.samples):
            if sample.source_file == source_name:
                self.builder_sample_combo.addItem(sample.name, index)

    def add_composite_part(self) -> None:
        sample_index = self.builder_sample_combo.currentData()
        if sample_index is None:
            QtWidgets.QMessageBox.warning(self, "No sample selected", "Choose a CSV source and sample first.")
            return
        profile_name = self.builder_profile_combo.currentText()
        self.composite_parts.append({"sample_index": int(sample_index), "profile": profile_name})
        self.refresh_composite_parts_list()

    def remove_selected_composite_part(self) -> None:
        row = self.composite_parts_list.currentRow()
        if row < 0 or row >= len(self.composite_parts):
            QtWidgets.QMessageBox.warning(self, "No segment selected", "Select a segment before removing it.")
            return
        self.composite_parts.pop(row)
        self.refresh_composite_parts_list()

    def clear_composite_parts(self) -> None:
        self.composite_parts.clear()
        self.refresh_composite_parts_list()

    def refresh_composite_parts_list(self) -> None:
        if not hasattr(self, "composite_parts_list"):
            return
        self.composite_parts_list.clear()
        for part in self.composite_parts:
            sample = self.samples[part["sample_index"]]
            self.composite_parts_list.addItem(f"{sample.name} -> {part['profile']}")

    def choose_composite_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(self.composite_color, self, "Choose composite line color")
        if not color.isValid():
            return
        self.composite_color = color
        self.composite_color_button.setStyleSheet(
            f"background-color: {self.composite_color.name()}; color: white; border: 1px solid #111827;"
        )

    def create_composite_sample(self) -> None:
        if not self.composite_parts:
            QtWidgets.QMessageBox.warning(self, "No segments", "Add at least one profile segment first.")
            return

        composite_rows_by_wavelength: dict[int, dict] = {}
        source_descriptions = []
        for part in self.composite_parts:
            sample = self.samples[part["sample_index"]]
            profile_name = part["profile"]
            indices = self.indices_for_profile_name(profile_name)
            selected_wavelengths = set(int(wl) for wl in WAVELENGTHS[indices])
            source_descriptions.append(f"{sample.name} [{profile_name}]")
            for row in sample.rows:
                wavelength = int(round(self.to_float(row.get("Golflengte (nm)", "nan"))))
                if wavelength in selected_wavelengths:
                    new_row = dict(row)
                    new_row["Sample naam"] = self.composite_name_edit.text().strip() or "Composite sample"
                    new_row["Composite source"] = " + ".join(source_descriptions)
                    composite_rows_by_wavelength[wavelength] = new_row

        if not composite_rows_by_wavelength:
            QtWidgets.QMessageBox.warning(self, "No data", "The selected segments contain no usable wavelengths.")
            return

        ordered_rows = [composite_rows_by_wavelength[int(wl)] for wl in WAVELENGTHS if int(wl) in composite_rows_by_wavelength]
        name = self.composite_name_edit.text().strip() or "Composite sample"
        self.samples.append(
            SpectrumSample(
                name=name,
                rows=ordered_rows,
                color=QtGui.QColor(self.composite_color),
                source_file="Samengesteld",
            )
        )
        self.rebuild_sample_controls()
        self.refresh_builder_source_combo()
        self.populate_table()
        self.refresh_all_plots()

    def indices_for_profile_name(self, profile_name: str) -> np.ndarray:
        profile_range = PROFILE_RANGES.get(profile_name)
        if profile_range is None:
            return np.arange(len(WAVELENGTHS))
        start, stop = profile_range
        return np.arange(start, stop)

    def profile_indices(self) -> np.ndarray:
        return self.indices_for_profile_name(self.profile_combo.currentText())

    def sample_xy(self, sample: SpectrumSample, column: str) -> tuple[np.ndarray, np.ndarray]:
        values_by_wavelength = {}
        for row in sample.rows:
            wavelength = self.to_float(row.get("Golflengte (nm)", ""))
            value = self.to_float(row.get(column, ""))
            if np.isfinite(wavelength) and np.isfinite(value):
                values_by_wavelength[int(round(wavelength))] = value

        indices = self.profile_indices()
        wavelengths = WAVELENGTHS[indices]
        values = np.array([values_by_wavelength.get(int(wl), np.nan) for wl in wavelengths], dtype=float)
        valid = np.isfinite(values)
        return wavelengths[valid], values[valid]

    def sample_value_by_wavelength(self, sample: SpectrumSample, column: str) -> dict[int, float]:
        values = {}
        for row in sample.rows:
            wavelength = self.to_float(row.get("Golflengte (nm)", ""))
            value = self.to_float(row.get(column, ""))
            if np.isfinite(wavelength) and np.isfinite(value):
                values[int(round(wavelength))] = value
        return values

    def visible_summary_rows(self) -> list[dict]:
        selected_wavelengths = [int(wl) for wl in WAVELENGTHS[self.profile_indices()]]
        rows = []
        for sample in self.samples:
            if not sample.visible:
                continue
            transmission = self.sample_value_by_wavelength(sample, "Genormaliseerde respons / transmissie (-)")
            absorbance = self.sample_value_by_wavelength(sample, "Absorbantie A = -log10(transmissie) (-)")
            sd = self.sample_value_by_wavelength(sample, "Standaarddeviatie spectrale respons (a.u.)")
            cv = self.sample_value_by_wavelength(sample, "Variatiecoefficient spectrale respons (%)")
            mean = self.sample_value_by_wavelength(sample, "Gemiddelde spectrale respons (a.u.)")
            for wavelength in selected_wavelengths:
                if any(wavelength in source for source in (transmission, absorbance, sd, cv, mean)):
                    rows.append({
                        "Sample": self.legend_label(sample),
                        "Wavelength (nm)": wavelength,
                        "Transmission (-)": transmission.get(wavelength, float("nan")),
                        "Absorbance (-)": absorbance.get(wavelength, float("nan")),
                        "Mean spectral response (a.u.)": mean.get(wavelength, float("nan")),
                        "Standard deviation (a.u.)": sd.get(wavelength, float("nan")),
                        "CV (%)": cv.get(wavelength, float("nan")),
                    })
        return rows

    def populate_rows_table(self, table: QtWidgets.QTableWidget | None, rows: list[dict]) -> None:
        if table is None:
            return
        if not rows:
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        headers = list(rows[0].keys())
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        for row_index, row in enumerate(rows):
            for col_index, header in enumerate(headers):
                value = row.get(header, "")
                if isinstance(value, float):
                    value = "" if not np.isfinite(value) else f"{value:.6g}"
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    def refresh_summary_tables(self) -> None:
        rows = self.visible_summary_rows()
        for table in self.metric_tables.values():
            self.populate_rows_table(table, rows)
        self.populate_rows_table(self.error_table, rows)
        self.populate_rows_table(self.interpolation_table, rows)

    def refresh_all_plots(self) -> None:
        for metric_key, plot in self.plot_widgets.items():
            self.draw_metric_plot(plot, metric_key)
        self.draw_error_plot()
        self.draw_interpolation_plot()
        self.refresh_summary_tables()

    def draw_metric_plot(self, plot: pg.PlotWidget, metric_key: str) -> None:
        plot.clear()
        meta = METRICS[metric_key]
        plot.setTitle(meta["title"], size=self.title_size, color="#111827")
        self.set_axis_label(plot, "left", meta["ylabel"])
        self.add_readable_legend(plot)
        plot.showGrid(x=self.show_grid_checkbox.isChecked(), y=self.show_grid_checkbox.isChecked(), alpha=0.08)
        plotted_values = []

        for sample in self.samples:
            if not sample.visible:
                continue
            x, y = self.sample_xy(sample, meta["column"])
            if x.size == 0:
                continue
            color = self.category_color_for_sample(sample, metric_key)
            plotted_values.extend(y.tolist())
            plot.plot(
                x,
                y,
                pen=pg.mkPen(color, width=self.line_width),
                symbol="o" if self.show_points_checkbox.isChecked() else None,
                symbolSize=self.marker_size,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(color),
                name=self.legend_label(sample),
            )
        self.apply_graph_scale(plot, plotted_values, metric_key, metric_key)

    def draw_error_plot(self, plot: pg.PlotWidget | None = None) -> None:
        plot = plot or self.error_plot
        if plot is None:
            return
        plot.clear()
        plot.setTitle("Mean spectral response with SD band", size=self.title_size, color="#111827")
        self.set_axis_label(plot, "left", "Spectral response (a.u.)")
        self.add_readable_legend(plot)
        plot.showGrid(x=self.show_grid_checkbox.isChecked(), y=self.show_grid_checkbox.isChecked(), alpha=0.08)
        plotted_values = []

        for sample in self.samples:
            if not sample.visible:
                continue
            x, y = self.sample_xy(sample, "Gemiddelde spectrale respons (a.u.)")
            _x_sd, sd = self.sample_xy(sample, "Standaarddeviatie spectrale respons (a.u.)")
            if x.size == 0 or sd.size != x.size:
                continue
            color = self.category_color_for_sample(sample, "error")
            plotted_values.extend((y + sd).tolist())
            plotted_values.extend((y - sd).tolist())
            fill_color = QtGui.QColor(color)
            fill_color.setAlpha(55)
            upper = plot.plot(x, y + sd, pen=None)
            lower = plot.plot(x, y - sd, pen=None)
            fill = pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(fill_color))
            plot.addItem(fill)
            plot.plot(
                x,
                y,
                pen=pg.mkPen(color, width=self.line_width),
                symbol="o",
                symbolSize=self.marker_size,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(color),
                name=self.legend_label(sample),
            )
        self.apply_graph_scale(plot, plotted_values, "error")

    def draw_interpolation_plot(self, plot: pg.PlotWidget | None = None) -> None:
        plot = plot or self.interpolation_plot
        if plot is None:
            return
        plot.clear()
        plot.setTitle("Interpolated transmission", size=self.title_size, color="#111827")
        self.set_axis_label(plot, "left", "Transmission (-)")
        self.add_readable_legend(plot)
        plot.showGrid(x=self.show_grid_checkbox.isChecked(), y=self.show_grid_checkbox.isChecked(), alpha=0.08)
        plotted_values = []

        for sample in self.samples:
            if not sample.visible:
                continue
            x, y = self.sample_xy(sample, "Genormaliseerde respons / transmissie (-)")
            if x.size < 2:
                continue
            color = self.category_color_for_sample(sample, "interpolation")
            dense_x = np.linspace(float(x.min()), float(x.max()), 400)
            dense_y = np.interp(dense_x, x, y)
            plotted_values.extend(dense_y.tolist())
            plot.plot(dense_x, dense_y, pen=pg.mkPen(color, width=self.line_width), name=f"{self.legend_label(sample)} interpolation")
            if self.show_points_checkbox.isChecked():
                plot.plot(x, y, pen=None, symbol="o", symbolSize=self.marker_size, symbolBrush=pg.mkBrush(color))
        self.apply_graph_scale(plot, plotted_values, "interpolation", "transmission")

    def value_at_wavelength(self, sample: SpectrumSample, column: str, wavelength: int) -> float:
        for row in sample.rows:
            row_wavelength_value = self.to_float(row.get("Golflengte (nm)", "nan"))
            if not np.isfinite(row_wavelength_value):
                continue
            row_wavelength = int(round(row_wavelength_value))
            if row_wavelength == wavelength:
                return self.to_float(row.get(column, "nan"))
        return float("nan")

    def calculate_linearity(self) -> None:
        if not self.samples:
            QtWidgets.QMessageBox.warning(self, "Linearity", "Load CSV data first.")
            return

        metric_label = self.linearity_metric_combo.currentText()
        column = LINEARITY_METRICS[metric_label]
        wavelength = int(self.linearity_wavelength_combo.currentData())
        unit = self.linearity_unit_edit.text().strip() or "concentration"

        rows = []
        for sample, edit in zip(self.samples, self.linearity_concentration_edits):
            concentration = self.to_float(edit.text())
            response = self.value_at_wavelength(sample, column, wavelength)
            if np.isfinite(concentration) and np.isfinite(response):
                rows.append({
                    "Sample": sample.name,
                    "Concentration": concentration,
                    "Eenheid": unit,
                    "Wavelength (nm)": wavelength,
                    "Parameter": metric_label,
                    "Respons": response,
                })

        if len(rows) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Linearity",
                "Enter at least 2 points. For the thesis, 3 concentrations plus a blank/reference zero point is better.",
            )
            return

        rows.sort(key=lambda row: row["Concentration"])
        x = np.array([row["Concentration"] for row in rows], dtype=float)
        y = np.array([row["Respons"] for row in rows], dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        for row, fit_value in zip(rows, fitted):
            row["Fitted response"] = float(fit_value)
            row["Residual"] = float(row["Respons"] - fit_value)
            row["Slope"] = float(slope)
            row["Intercept"] = float(intercept)
            row["R2"] = float(r_squared)

        self.linearity_last_rows = rows
        self.draw_linearity_plot(x, y, slope, intercept, r_squared, metric_label, wavelength, unit)
        self.populate_rows_table(self.linearity_table, rows)

        has_blank_point = bool(np.any(np.isclose(x, 0.0)))
        quality_note = "Use this as an indicative linearity test."
        if len(rows) >= 5:
            quality_note = "This is stronger than a 3-point test and more defensible as a calibration curve."
        elif len(rows) >= 4 and has_blank_point:
            quality_note = "This is a good practical setup: 3 concentrations plus a blank/reference zero point."
        elif len(rows) == 3:
            quality_note = "This shows a trend, but preferably add the blank/reference as a zero point."

        self.linearity_result_label.setText(
            f"Result for {metric_label} at {wavelength} nm: "
            f"y = {slope:.6g} x + {intercept:.6g}, R2 = {r_squared:.5f}. {quality_note}"
        )

    def draw_linearity_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        slope: float,
        intercept: float,
        r_squared: float,
        metric_label: str,
        wavelength: int,
        unit: str,
    ) -> None:
        if self.linearity_plot is None:
            return
        plot = self.linearity_plot
        plot.clear()
        self.add_readable_legend(plot)
        plot.setTitle(f"Linearity at {wavelength} nm - {metric_label}", size=self.title_size, color="#111827")
        self.set_axis_label(plot, "bottom", f"Concentration ({unit})")
        self.set_axis_label(plot, "left", metric_label)
        plot.showGrid(x=self.show_grid_checkbox.isChecked(), y=self.show_grid_checkbox.isChecked(), alpha=0.08)
        linearity_color = self.graph_setting("linearity").get("color", QtGui.QColor("#111827"))

        plot.plot(
            x,
            y,
            pen=None,
            symbol="o",
            symbolSize=self.marker_size + 2,
            symbolBrush=pg.mkBrush(linearity_color),
            symbolPen=pg.mkPen(linearity_color),
            name="Measurement points",
        )

        x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_fit = slope * x_fit + intercept
        plot.plot(x_fit, y_fit, pen=pg.mkPen(linearity_color, width=self.line_width), name=f"Linear fit, R2={r_squared:.4f}")
        self.apply_graph_scale(plot, y.tolist() + y_fit.tolist(), "linearity")

    def export_linearity_png(self) -> None:
        if self.linearity_plot is None or not self.linearity_last_rows:
            QtWidgets.QMessageBox.warning(self, "Linearity PNG", "Calculate linearity first.")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export linearity as PNG",
            str(Path(__file__).resolve().parent / "linearity_r2.png"),
            "PNG files (*.png)",
        )
        if not file_name:
            return
        try:
            self.export_linearity_publication_png(file_name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Linearity PNG", f"Export failed:\n{exc}")

    def export_linearity_csv(self) -> None:
        if not self.linearity_last_rows:
            QtWidgets.QMessageBox.warning(self, "Linearity CSV", "Calculate linearity first.")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export linearity as CSV",
            str(Path(__file__).resolve().parent / "linearity_r2.csv"),
            "CSV files (*.csv)",
        )
        if not file_name:
            return

        headers = [
            "Sample",
            "Concentration",
            "Eenheid",
            "Wavelength (nm)",
            "Parameter",
            "Respons",
            "Fitted response",
            "Residual",
            "Slope",
            "Intercept",
            "R2",
        ]
        with Path(file_name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, delimiter=";")
            writer.writeheader()
            writer.writerows(self.linearity_last_rows)

    def open_spectrometer_reference_csv(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open blank-corrected spectrometer CSV",
            str(Path(__file__).resolve().parent),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not file_name:
            return

        try:
            self.spectrometer_reference_rows = self.load_spectrometer_reference_csv(Path(file_name))
        except (OSError, csv.Error, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Spectrometer CSV loading failed", str(exc))
            return

        self.spectrometer_reference_path = Path(file_name)
        self.spectrometer_file_label.setText(f"Reference loaded: {self.spectrometer_reference_path.name}")
        self.spectrometer_result_label.setText(
            f"{len(self.spectrometer_reference_rows)} wavelengths loaded from {self.spectrometer_reference_path.name}."
        )

    def load_spectrometer_reference_csv(self, path: Path) -> dict[int, dict]:
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            sample = handle.read(2048)
            delimiter = ";" if sample.count(";") >= sample.count(",") else ","
            handle.seek(0)
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                raise ValueError("Spectrometer CSV has no header.")
            rows = list(reader)

        if not rows:
            raise ValueError("Spectrometer CSV contains no data.")

        wavelength_column = self.find_column(reader.fieldnames, ["wavelength"])
        transmission_column = self.find_column(reader.fieldnames, ["blank-corrected transmission"])
        if transmission_column is None:
            transmission_column = self.find_column(reader.fieldnames, ["corrected transmission"])
        absorbance_column = self.find_column(reader.fieldnames, ["absorbance"])

        if wavelength_column is None:
            raise ValueError("No wavelength column found in spectrometer CSV.")
        if transmission_column is None and absorbance_column is None:
            raise ValueError("No corrected transmission or absorbance column found in spectrometer CSV.")

        values_by_wavelength = {}
        for row in rows:
            wavelength_value = self.to_float(row.get(wavelength_column, "nan"))
            if not np.isfinite(wavelength_value):
                continue
            wavelength = int(round(wavelength_value))
            transmission_percent = self.to_float(row.get(transmission_column, "nan")) if transmission_column else float("nan")
            absorbance = self.to_float(row.get(absorbance_column, "nan")) if absorbance_column else float("nan")
            values_by_wavelength[wavelength] = {
                "transmission_fraction": transmission_percent / 100 if np.isfinite(transmission_percent) else float("nan"),
                "transmission_percent": transmission_percent,
                "absorbance": absorbance,
            }
        return values_by_wavelength

    def find_column(self, columns: list[str], required_parts: list[str]) -> str | None:
        for column in columns:
            lowered = column.lower()
            if all(part.lower() in lowered for part in required_parts):
                return column
        return None

    def compare_with_spectrometer(self) -> None:
        if not self.spectrometer_reference_rows:
            QtWidgets.QMessageBox.warning(self, "Spectrometer comparison", "Load a corrected spectrometer CSV first.")
            return

        sample_index = self.spectrometer_sample_combo.currentData()
        if sample_index is None or sample_index < 0 or sample_index >= len(self.samples):
            QtWidgets.QMessageBox.warning(self, "Spectrometer comparison", "Choose an AS7265x sample first.")
            return

        sample = self.samples[int(sample_index)]
        metric = self.spectrometer_metric_combo.currentText()
        as_column = LINEARITY_METRICS[metric]
        reference_key = "transmission_fraction" if metric.startswith("Transmission") else "absorbance"

        comparison_rows = []
        for wavelength in WAVELENGTHS:
            wavelength_int = int(wavelength)
            if wavelength_int not in self.spectrometer_reference_rows:
                continue
            as_value = self.value_at_wavelength(sample, as_column, wavelength_int)
            reference_value = self.spectrometer_reference_rows[wavelength_int].get(reference_key, float("nan"))
            if not np.isfinite(as_value) or not np.isfinite(reference_value):
                continue

            error = as_value - reference_value
            absolute_error = abs(error)
            relative_error_percent = absolute_error / abs(reference_value) * 100 if reference_value != 0 else float("nan")
            comparison_rows.append({
                "Wavelength (nm)": wavelength_int,
                "AS7265x value": as_value,
                "Reference spectrometer value": reference_value,
                "Difference AS7265x - spectrometer": error,
                "Absolute error": absolute_error,
                "Relative error (%)": relative_error_percent,
                "Parameter": metric,
                "AS7265x sample": sample.name,
                "Spectrometer CSV": self.spectrometer_reference_path.name if self.spectrometer_reference_path else "",
            })

        if len(comparison_rows) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Spectrometer comparison",
                "Fewer than 2 overlapping wavelengths were found between AS7265x and the reference spectrometer.",
            )
            return

        self.spectrometer_comparison_rows = comparison_rows
        self.draw_spectrometer_comparison()
        self.populate_spectrometer_table()

        reference = np.array([row["Reference spectrometer value"] for row in comparison_rows], dtype=float)
        as_values = np.array([row["AS7265x value"] for row in comparison_rows], dtype=float)
        if np.std(reference) > 0 and np.std(as_values) > 0:
            r_squared = float(np.corrcoef(reference, as_values)[0, 1] ** 2)
        else:
            r_squared = float("nan")
        mae = float(np.mean([row["Absolute error"] for row in comparison_rows]))
        mean_relative_error = float(np.nanmean([row["Relative error (%)"] for row in comparison_rows]))

        self.spectrometer_result_label.setText(
            f"{len(comparison_rows)} overlapping AS7265x channels used. "
            f"Mean absolute error = {mae:.5g}; mean relative error = {mean_relative_error:.3g}%; "
            f"agreement/R2 indication = {r_squared:.5f}. "
            "Use this mainly to evaluate the spectral trend, not as perfect point-by-point validation."
        )

    def draw_spectrometer_comparison(self) -> None:
        if self.spectrometer_compare_plot is None or self.spectrometer_error_plot is None:
            return
        wavelengths = np.array([row["Wavelength (nm)"] for row in self.spectrometer_comparison_rows], dtype=float)
        as_values = np.array([row["AS7265x value"] for row in self.spectrometer_comparison_rows], dtype=float)
        reference = np.array([row["Reference spectrometer value"] for row in self.spectrometer_comparison_rows], dtype=float)
        errors = np.array([row["Absolute error"] for row in self.spectrometer_comparison_rows], dtype=float)
        metric = self.spectrometer_metric_combo.currentText()
        comparison_color = self.graph_setting("spectrometer").get("color", QtGui.QColor("#111827"))
        error_color = self.graph_setting("spectrometer_error").get("color", QtGui.QColor("#2563eb"))

        self.spectrometer_compare_plot.clear()
        self.add_readable_legend(self.spectrometer_compare_plot)
        self.spectrometer_compare_plot.setTitle(f"AS7265x versus reference spectrometer - {metric}", size=self.title_size, color="#111827")
        self.set_axis_label(self.spectrometer_compare_plot, "left", metric)
        self.spectrometer_compare_plot.showGrid(x=True, y=True, alpha=0.08)
        self.spectrometer_compare_plot.plot(
            wavelengths,
            reference,
            pen=pg.mkPen("#dc2626", width=self.line_width),
            symbol="s",
            symbolSize=self.marker_size,
            symbolBrush=pg.mkBrush("#dc2626"),
            name="Reference spectrometer",
        )
        self.spectrometer_compare_plot.plot(
            wavelengths,
            as_values,
            pen=pg.mkPen(comparison_color, width=self.line_width),
            symbol="o",
            symbolSize=self.marker_size,
            symbolBrush=pg.mkBrush(comparison_color),
            name="AS7265x",
        )
        self.apply_graph_scale(
            self.spectrometer_compare_plot,
            reference.tolist() + as_values.tolist(),
            "spectrometer",
        )

        self.spectrometer_error_plot.clear()
        self.spectrometer_error_plot.setTitle("Absolute error per channel", size=self.title_size, color="#111827")
        self.set_axis_label(self.spectrometer_error_plot, "left", f"Absolute error ({metric})")
        self.spectrometer_error_plot.showGrid(x=True, y=True, alpha=0.08)
        self.spectrometer_error_plot.plot(
            wavelengths,
            errors,
            pen=pg.mkPen(error_color, width=self.line_width),
            symbol="o",
            symbolSize=self.marker_size,
            symbolBrush=pg.mkBrush(error_color),
        )
        self.apply_graph_scale(self.spectrometer_error_plot, errors.tolist(), "spectrometer_error")

    def populate_spectrometer_table(self) -> None:
        if not hasattr(self, "spectrometer_table"):
            return
        headers = [
            "Wavelength (nm)",
            "AS7265x value",
            "Reference spectrometer value",
            "Difference AS7265x - spectrometer",
            "Absolute error",
            "Relative error (%)",
        ]
        self.spectrometer_table.setRowCount(len(self.spectrometer_comparison_rows))
        self.spectrometer_table.setColumnCount(len(headers))
        self.spectrometer_table.setHorizontalHeaderLabels(headers)
        for row_index, row in enumerate(self.spectrometer_comparison_rows):
            for col_index, header in enumerate(headers):
                value = row.get(header, "")
                if isinstance(value, float):
                    value = f"{value:.6g}"
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.spectrometer_table.setItem(row_index, col_index, item)
        self.spectrometer_table.resizeColumnsToContents()

    def export_spectrometer_comparison_png(self) -> None:
        if not self.spectrometer_comparison_rows or self.spectrometer_compare_plot is None:
            QtWidgets.QMessageBox.warning(self, "Spectrometer PNG", "Create a spectrometer comparison first.")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export spectrometer comparison as PNG",
            str(Path(__file__).resolve().parent / "spectrometer_comparison.png"),
            "PNG files (*.png)",
        )
        if not file_name:
            return
        try:
            self.export_spectrometer_comparison_publication_png(file_name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Spectrometer PNG", f"Export failed:\n{exc}")

    def export_spectrometer_comparison_csv(self) -> None:
        if not self.spectrometer_comparison_rows:
            QtWidgets.QMessageBox.warning(self, "Spectrometer CSV", "Create a spectrometer comparison first.")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export spectrometer error table as CSV",
            str(Path(__file__).resolve().parent / "spectrometer_comparison_errors.csv"),
            "CSV files (*.csv)",
        )
        if not file_name:
            return
        headers = list(self.spectrometer_comparison_rows[0].keys())
        with Path(file_name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, delimiter=";")
            writer.writeheader()
            writer.writerows(self.spectrometer_comparison_rows)

    def apply_plot_range(self, plot: pg.PlotWidget) -> None:
        indices = self.profile_indices()
        if indices.size:
            plot.setXRange(float(WAVELENGTHS[indices[0]]) - 10, float(WAVELENGTHS[indices[-1]]) + 10, padding=0)
        plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    def apply_thesis_y_range(self, plot: pg.PlotWidget, values: list[float], metric_key: str | None = None) -> None:
        finite = np.array([value for value in values if np.isfinite(value)], dtype=float)
        if finite.size == 0:
            plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            return
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        if metric_key == "transmission" and 0.85 <= ymin <= ymax <= 1.15:
            ymin = max(0.95, ymin - 0.01)
            ymax = min(1.05, ymax + 0.01)
            if ymax - ymin < 0.03:
                center = (ymin + ymax) / 2
                ymin, ymax = center - 0.02, center + 0.02
            plot.setYRange(ymin, ymax, padding=0)
            return
        span = ymax - ymin
        if span <= 0:
            span = max(abs(ymax) * 0.05, 0.05)
        margin = span * 0.12
        plot.setYRange(ymin - margin, ymax + margin, padding=0)

    def populate_table(self) -> None:
        all_rows = [row for sample in self.samples for row in sample.rows]
        if not all_rows:
            self.table.clear()
            return
        headers = list(all_rows[0].keys())
        self.table.setRowCount(len(all_rows))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for row_index, row in enumerate(all_rows):
            for col_index, header in enumerate(headers):
                item = QtWidgets.QTableWidgetItem(row.get(header, ""))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row_index, col_index, item)
        self.table.resizeColumnsToContents()

    def current_plot_widget(self) -> pg.PlotWidget | None:
        current_widget = self.tabs.currentWidget()
        if isinstance(current_widget, pg.PlotWidget):
            return current_widget
        if current_widget is None:
            return None
        plots = current_widget.findChildren(pg.PlotWidget)
        return plots[0] if plots else None

    def current_table_widget(self) -> QtWidgets.QTableWidget | None:
        current_widget = self.tabs.currentWidget()
        if isinstance(current_widget, QtWidgets.QTableWidget):
            return current_widget
        if current_widget is None:
            return None
        tables = current_widget.findChildren(QtWidgets.QTableWidget)
        return tables[0] if tables else None

    def current_metric_key(self) -> str | None:
        current_plot = self.current_plot_widget()
        for metric_key, plot in self.plot_widgets.items():
            if current_plot is plot:
                return metric_key
        return None

    def prepare_publication_axis(self, ax, title: str, x_label: str, y_label: str) -> None:
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=22, color="#111827")
        ax.set_xlabel(x_label, fontsize=11, fontweight="semibold", labelpad=7, color="#111827")
        ax.set_ylabel(y_label, fontsize=11, fontweight="semibold", labelpad=7, color="#111827")
        ax.tick_params(axis="both", which="major", labelsize=9, width=0.75, length=3.5, colors="#111827")
        ax.grid(True, which="major", color="#cbd5e1", linewidth=0.5, alpha=0.34)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#111827")
        ax.spines["bottom"].set_color("#111827")
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

    def apply_publication_y_range(self, ax, values: list[float], metric_key: str | None = None) -> None:
        finite = np.array([value for value in values if np.isfinite(value)], dtype=float)
        if finite.size == 0:
            return
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        if metric_key == "transmission" and 0.85 <= ymin <= ymax <= 1.15:
            center = 1.0 if abs(((ymin + ymax) / 2) - 1.0) < 0.04 else (ymin + ymax) / 2
            half_span = max((ymax - ymin) * 0.8, 0.025)
            ax.set_ylim(center - half_span, center + half_span)
            return
        span = ymax - ymin
        if span <= 0:
            span = max(abs(ymax) * 0.05, 0.05)
        margin = span * 0.14
        ax.set_ylim(ymin - margin, ymax + margin)

    def add_publication_legend(self, ax, handles_count: int) -> None:
        if handles_count == 0:
            return
        columns = min(3, max(1, handles_count))
        legend = ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=columns,
            frameon=False,
            fontsize=8,
            handlelength=2.0,
            columnspacing=1.4,
        )
        for text in legend.get_texts():
            text.set_color("#111827")

    def export_metric_publication_png(self, file_name: str, metric_key: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        meta = METRICS[metric_key]
        fig, ax = plt.subplots(figsize=self.publication_figsize, dpi=self.publication_dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        plotted_values: list[float] = []
        handles_count = 0
        for sample in self.samples:
            if not sample.visible:
                continue
            x, y = self.sample_xy(sample, meta["column"])
            if x.size == 0:
                continue
            color = self.category_color_for_sample(sample, metric_key).name()
            plotted_values.extend(y.tolist())
            ax.plot(
                x,
                y,
                color=color,
                linewidth=1.2,
                marker="o",
                markersize=3.3,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=self.legend_label(sample),
            )
            handles_count += 1

        self.apply_publication_scale(ax, plotted_values, metric_key, metric_key)
        self.prepare_publication_axis(ax, meta["title"], "Wavelength (nm)", meta["ylabel"])
        self.add_publication_legend(ax, handles_count)
        fig.tight_layout()
        fig.savefig(file_name, dpi=self.publication_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def export_error_publication_png(self, file_name: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.publication_figsize, dpi=self.publication_dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        plotted_values: list[float] = []
        handles_count = 0
        for sample in self.samples:
            if not sample.visible:
                continue
            x, y = self.sample_xy(sample, "Gemiddelde spectrale respons (a.u.)")
            _x_sd, sd = self.sample_xy(sample, "Standaarddeviatie spectrale respons (a.u.)")
            if x.size == 0 or sd.size != x.size:
                continue
            color = self.category_color_for_sample(sample, "error").name()
            plotted_values.extend((y + sd).tolist())
            plotted_values.extend((y - sd).tolist())
            ax.fill_between(
                x, y - sd, y + sd,
                color=color,
                alpha=0.18,
            )
            ax.plot(
                x, y,
                color=color,
                linewidth=1.2,
                marker="o",
                markersize=3.3,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=self.legend_label(sample),
            )
            handles_count += 1
        self.apply_publication_scale(ax, plotted_values, "error")
        self.prepare_publication_axis(ax, "Mean spectral response with SD band", "Wavelength (nm)", "Spectral response (a.u.)")
        self.add_publication_legend(ax, handles_count)
        fig.tight_layout()
        fig.savefig(file_name, dpi=self.publication_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def export_error_subplot_publication_png(self, file_name: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_mean, ax_sd) = plt.subplots(
            2, 1,
            figsize=(self.publication_figsize[0], self.publication_figsize[1] * 1.65),
            dpi=self.publication_dpi,
            sharex=True,
        )
        fig.patch.set_facecolor("white")
        ax_mean.set_facecolor("white")
        ax_sd.set_facecolor("white")

        plotted_mean: list[float] = []
        plotted_sd: list[float] = []
        handles_count = 0

        for sample in self.samples:
            if not sample.visible:
                continue
            x, y = self.sample_xy(sample, "Gemiddelde spectrale respons (a.u.)")
            _x_sd, sd = self.sample_xy(sample, "Standaarddeviatie spectrale respons (a.u.)")
            if x.size == 0 or sd.size != x.size:
                continue
            plotted_mean.extend(y.tolist())
            plotted_sd.extend(sd.tolist())
            label = self.legend_label(sample)
            mean_color = self.category_color_for_sample(sample, "error").name()
            sd_color = self.category_color_for_sample(sample, "error_sd_subplot").name()

            ax_mean.fill_between(x, y - sd, y + sd, color=mean_color, alpha=0.18)
            ax_mean.plot(
                x, y,
                color=mean_color, linewidth=1.2, marker="o", markersize=3.3,
                markerfacecolor=mean_color, markeredgecolor="white", markeredgewidth=0.45,
                label=label,
            )
            ax_sd.plot(
                x, sd,
                color=sd_color, linewidth=1.2, marker="o", markersize=3.3,
                markerfacecolor=sd_color, markeredgecolor="white", markeredgewidth=0.45,
                label=label,
            )
            handles_count += 1

        self.apply_publication_scale(ax_mean, plotted_mean, "error")
        self.apply_publication_scale(ax_sd, plotted_sd, "error_sd_subplot")
        self.prepare_publication_axis(
            ax_mean, "Mean spectral response with SD band", "", "Spectral response (a.u.)",
        )
        self.prepare_publication_axis(
            ax_sd, "Standard deviation per channel", "Wavelength (nm)", "Standard deviation (a.u.)",
        )
        self.add_publication_legend(ax_mean, handles_count)
        fig.tight_layout()
        fig.savefig(file_name, dpi=self.publication_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def export_error_subplot_png(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export mean + SD subplot as PNG",
            str(Path(__file__).resolve().parent / "mean_sd_subplot.png"),
            "PNG files (*.png)",
        )
        if not file_name:
            return
        try:
            self.export_error_subplot_publication_png(file_name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "SD subplot PNG", f"Export failed:\n{exc}")

    def export_interpolation_publication_png(self, file_name: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.publication_figsize, dpi=self.publication_dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        plotted_values: list[float] = []
        handles_count = 0
        for sample in self.samples:
            if not sample.visible:
                continue
            x, y = self.sample_xy(sample, "Genormaliseerde respons / transmissie (-)")
            if x.size < 2:
                continue
            color = self.category_color_for_sample(sample, "interpolation").name()
            dense_x = np.linspace(float(x.min()), float(x.max()), 400)
            dense_y = np.interp(dense_x, x, y)
            plotted_values.extend(dense_y.tolist())
            ax.plot(dense_x, dense_y, color=color, linewidth=1.2, label=f"{self.legend_label(sample)} interpolation")
            ax.scatter(x, y, color=color, s=10, edgecolors="white", linewidths=0.4, zorder=3)
            handles_count += 1
        self.apply_publication_scale(ax, plotted_values, "interpolation", "transmission")
        self.prepare_publication_axis(ax, "Interpolated transmission", "Wavelength (nm)", "Transmission (-)")
        self.add_publication_legend(ax, handles_count)
        fig.tight_layout()
        fig.savefig(file_name, dpi=self.publication_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def export_linearity_publication_png(self, file_name: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.linearity_last_rows:
            return

        rows = self.linearity_last_rows
        x = np.array([row["Concentration"] for row in rows], dtype=float)
        y = np.array([row["Respons"] for row in rows], dtype=float)
        slope = rows[0]["Slope"]
        intercept = rows[0]["Intercept"]
        r_squared = rows[0]["R2"]
        metric_label = rows[0]["Parameter"]
        wavelength = rows[0]["Wavelength (nm)"]
        unit = rows[0]["Eenheid"]

        x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_fit = slope * x_fit + intercept
        linearity_color = self.graph_setting("linearity").get("color", QtGui.QColor("#111827")).name()

        fig, ax = plt.subplots(figsize=self.publication_figsize, dpi=self.publication_dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.scatter(x, y, color=linearity_color, s=28, zorder=3, label="Measurement points")
        ax.plot(x_fit, y_fit, color=linearity_color, linewidth=1.2, label=f"Linear fit, R²={r_squared:.4f}")

        self.apply_publication_scale(ax, y.tolist() + y_fit.tolist(), "linearity")
        self.prepare_publication_axis(
            ax,
            f"Linearity at {wavelength} nm – {metric_label}",
            f"Concentration ({unit})",
            metric_label,
        )
        self.add_publication_legend(ax, 2)
        fig.tight_layout()
        fig.savefig(file_name, dpi=self.publication_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def export_spectrometer_comparison_publication_png(self, file_name: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.spectrometer_comparison_rows:
            return

        wavelengths = np.array([row["Wavelength (nm)"] for row in self.spectrometer_comparison_rows], dtype=float)
        as_values = np.array([row["AS7265x value"] for row in self.spectrometer_comparison_rows], dtype=float)
        reference = np.array([row["Reference spectrometer value"] for row in self.spectrometer_comparison_rows], dtype=float)
        errors = np.array([row["Absolute error"] for row in self.spectrometer_comparison_rows], dtype=float)
        metric = self.spectrometer_comparison_rows[0]["Parameter"]
        comparison_color = self.graph_setting("spectrometer").get("color", QtGui.QColor("#111827")).name()
        error_color = self.graph_setting("spectrometer_error").get("color", QtGui.QColor("#2563eb")).name()

        fig, ax = plt.subplots(figsize=self.publication_figsize, dpi=self.publication_dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.plot(
            wavelengths, reference,
            color="#dc2626", linewidth=1.2, marker="s", markersize=3.3,
            markerfacecolor="#dc2626", markeredgecolor="white", markeredgewidth=0.45,
            label="Reference spectrometer",
        )
        ax.plot(
            wavelengths, as_values,
            color=comparison_color, linewidth=1.2, marker="o", markersize=3.3,
            markerfacecolor=comparison_color, markeredgecolor="white", markeredgewidth=0.45,
            label="AS7265x",
        )
        self.apply_publication_scale(ax, reference.tolist() + as_values.tolist(), "spectrometer")
        self.prepare_publication_axis(
            ax, f"AS7265x versus reference spectrometer – {metric}", "Wavelength (nm)", metric,
        )
        self.add_publication_legend(ax, 2)
        fig.tight_layout()
        fig.savefig(file_name, dpi=self.publication_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        path = Path(file_name)
        error_path = str(path.with_name(f"{path.stem}_absolute_error{path.suffix}"))
        fig, ax = plt.subplots(figsize=self.publication_figsize, dpi=self.publication_dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.plot(
            wavelengths, errors,
            color=error_color, linewidth=1.2, marker="o", markersize=3.3,
            markerfacecolor=error_color, markeredgecolor="white", markeredgewidth=0.45,
        )
        self.apply_publication_scale(ax, errors.tolist(), "spectrometer_error")
        self.prepare_publication_axis(
            ax, "Absolute error per channel", "Wavelength (nm)", f"Absolute error ({metric})",
        )
        self.add_publication_legend(ax, 0)
        fig.tight_layout()
        fig.savefig(error_path, dpi=self.publication_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def export_current_png(self) -> None:
        current_plot = self.current_plot_widget()
        if current_plot is None:
            QtWidgets.QMessageBox.warning(self, "PNG export", "Select a graph tab first.")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export graph as PNG",
            str(Path(__file__).resolve().parent / "spectro_graph.png"),
            "PNG files (*.png)",
        )
        if not file_name:
            return
        try:
            metric_key = self.current_metric_key()
            if metric_key is not None:
                self.export_metric_publication_png(file_name, metric_key)
            elif current_plot is self.error_plot:
                self.export_error_publication_png(file_name)   # shaded band export
            elif current_plot is self.interpolation_plot:
                self.export_interpolation_publication_png(file_name)
            elif current_plot is self.linearity_plot:
                self.export_linearity_publication_png(file_name)
            elif current_plot in (self.spectrometer_compare_plot, self.spectrometer_error_plot):
                self.export_spectrometer_comparison_publication_png(file_name)
            else:
                exporter = pg.exporters.ImageExporter(current_plot.plotItem)
                exporter.parameters()["width"] = self.export_width
                exporter.export(file_name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "PNG export", f"Publication export failed:\n{exc}")

    def export_current_table_csv(self) -> None:
        current_table = self.current_table_widget()
        if current_table is None or current_table.columnCount() == 0:
            QtWidgets.QMessageBox.warning(self, "Table export", "Select a tab with table data first.")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export current table as CSV",
            str(Path(__file__).resolve().parent / "spectro_current_table.csv"),
            "CSV files (*.csv)",
        )
        if not file_name:
            return
        headers = [
            current_table.horizontalHeaderItem(col).text()
            if current_table.horizontalHeaderItem(col) is not None else f"Column {col + 1}"
            for col in range(current_table.columnCount())
        ]
        with Path(file_name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(headers)
            for row in range(current_table.rowCount()):
                writer.writerow([
                    current_table.item(row, col).text() if current_table.item(row, col) is not None else ""
                    for col in range(current_table.columnCount())
                ])

    def export_filtered_csv(self) -> None:
        if not self.samples:
            QtWidgets.QMessageBox.warning(self, "CSV export", "Open a CSV file first.")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export filtered CSV",
            str(Path(__file__).resolve().parent / "spectro_filtered.csv"),
            "CSV files (*.csv)",
        )
        if not file_name:
            return

        selected_wavelengths = set(int(wl) for wl in WAVELENGTHS[self.profile_indices()])
        visible_rows = []
        for sample in self.samples:
            if not sample.visible:
                continue
            for row in sample.rows:
                wavelength_value = self.to_float(row.get("Golflengte (nm)", "nan"))
                if not np.isfinite(wavelength_value):
                    continue
                wavelength = int(round(wavelength_value))
                if wavelength in selected_wavelengths:
                    visible_rows.append(row)

        if not visible_rows:
            QtWidgets.QMessageBox.warning(self, "CSV export", "No visible data to export.")
            return

        headers = list(visible_rows[0].keys())
        with Path(file_name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, delimiter=";")
            writer.writeheader()
            writer.writerows(visible_rows)

    def to_float(self, value: str) -> float:
        try:
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return float("nan")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = CsvSpectrumAnalyzer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
